/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/converter_ext/gemma.cc
 * \brief Implementation of the Gemma tool calling converter.
 */
#include <picojson.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "../json_schema_converter_ext.h"
#include "../support/logging.h"

namespace xgrammar {

namespace {

bool IsIdentifier(const std::string& text) {
  if (text.empty() || (text[0] >= '0' && text[0] <= '9')) {
    return false;
  }
  return std::all_of(text.begin(), text.end(), [](unsigned char character) {
    return (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') ||
           (character >= '0' && character <= '9') || character == '_';
  });
}

// Identifier characters, minus the ones a child branch of the key trie already consumes. The
// first character of a key cannot be a digit, so the caller decides whether digits are allowed.
template <typename Children>
std::vector<GrammarBuilder::CharacterClassElement> IdentifierRangesExcluding(
    const Children& children, bool allow_digits
) {
  auto allowed = [&](int32_t character) {
    bool is_identifier_character = (character >= 'a' && character <= 'z') ||
                                   (character >= 'A' && character <= 'Z') || character == '_' ||
                                   (allow_digits && character >= '0' && character <= '9');
    return is_identifier_character && children.count(static_cast<uint8_t>(character)) == 0;
  };

  std::vector<GrammarBuilder::CharacterClassElement> ranges;
  for (int32_t character = '0'; character <= 'z'; ++character) {
    if (!allowed(character)) {
      continue;
    }
    int32_t range_start = character;
    while (character + 1 <= 'z' && allowed(character + 1)) {
      ++character;
    }
    ranges.push_back({range_start, character});
  }
  return ranges;
}

std::string ToLowerASCII(const std::string& text) {
  std::string lowered = text;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char byte) {
    return static_cast<char>(byte >= 'A' && byte <= 'Z' ? byte - 'A' + 'a' : byte);
  });
  return lowered;
}

}  // namespace

const std::string GemmaToolCallingConverter::kGemmaStringDelim = "<|\"|>";
const std::string GemmaToolCallingConverter::kGemmaStringContent = "gemma_string_content";
const std::string GemmaToolCallingConverter::kGemmaVariableName = "gemma_variable_name";

GemmaToolCallingConverter::GemmaToolCallingConverter(
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool any_whitespace,
    std::optional<int> max_whitespace_cnt,
    RefResolver ref_resolver,
    bool any_order
)
    : JSONSchemaConverter(
          indent, separators, any_whitespace, max_whitespace_cnt, ref_resolver, any_order
      ) {}

void GemmaToolCallingConverter::AddBasicRules() {
  JSONSchemaConverter::AddBasicRules({kGemmaStringContent, kGemmaVariableName});

  // A tool-call marker inside a string desyncs every downstream parser, and the chat template
  // escapes nothing, so such a string cannot be produced in the first place.
  builder_.UpdateRuleBody(
      kGemmaStringContent,
      TagDispatch(
          /*loop_after_dispatch=*/false, {kGemmaStringDelim, "<|tool_call>", "<tool_call|>"}
      )
  );
  builder_.UpdateRuleBody(
      kGemmaVariableName,
      Sequence(
          {builder_.AddCharacterClass({{'a', 'z'}, {'A', 'Z'}, {'_', '_'}}),
           builder_.AddCharacterClassStar({{'a', 'z'}, {'A', 'Z'}, {'0', '9'}, {'_', '_'}})}
      )
  );
  // The base class writes basic_string as a JSON-quoted string directly rather than through
  // GenerateString, so the delimited form has to replace it after the base call. That leaves
  // basic_escape and basic_string_sub unreachable from root.
  builder_.UpdateRuleBody(kBasicString, GenerateString(StringSpec{}, kBasicString));
}

int32_t GemmaToolCallingConverter::GenerateString(
    const StringSpec& spec, const std::string& rule_name
) {
  int32_t delimiter = ByteString(kGemmaStringDelim);
  // A regex is matched against the raw content, so it is not JSON-escaped here. A pattern or
  // format that can generate the delimiter text closes the string early; callers must keep
  // patterns restrictive.
  if (spec.format.has_value()) {
    auto regex = JSONFormatToRegexPattern(*spec.format);
    if (regex.has_value()) {
      // The built-in format regexes use constructs that the FSM regex engine does not fully
      // support yet (e.g. quoted email local parts), so they keep the CFG expansion.
      return Sequence(
          {delimiter,
           RegexExpression(*regex, /*json_string=*/false, /*force_cfg_expansion=*/true),
           delimiter}
      );
    }
  }
  if (spec.pattern.has_value()) {
    return Sequence({delimiter, RegexExpression(*spec.pattern, /*json_string=*/false), delimiter});
  }
  if (spec.min_length != 0 || spec.max_length != -1) {
    // The counted characters are not restricted to exclude the delimiter text: the parser stops
    // at the first delimiter, so a bounded string that reaches it just ends there. The effect is
    // an extra parse, never a lost one - JSON style is loose in the same way.
    int32_t character = builder_.AddCharacterClass({{0, 0x10FFFF}});
    int32_t body = Repeat(rule_name + "_characters", character, spec.min_length, spec.max_length);
    return Sequence({delimiter, body, delimiter});
  }
  return Sequence({delimiter, RuleRef(kGemmaStringContent), delimiter});
}

int32_t GemmaToolCallingConverter::GenerateObject(
    const ObjectSpec& spec, const std::string& rule_name, bool need_brace
) {
  // patternProperties and propertyNames route their keys through GenerateString, which would
  // emit them delimited rather than bare.
  XGRAMMAR_CHECK(spec.pattern_properties.empty())
      << "gemma style does not support patternProperties";
  XGRAMMAR_CHECK(spec.property_names == nullptr) << "gemma style does not support propertyNames";
  // The chat template renders the arguments with dictsort, so the model emits the declared
  // properties key-sorted rather than in declaration order.
  ObjectSpec sorted_spec = spec;
  std::stable_sort(
      sorted_spec.properties.begin(),
      sorted_spec.properties.end(),
      [](const ObjectSpec::Property& lhs, const ObjectSpec::Property& rhs) {
        return ToLowerASCII(lhs.name) < ToLowerASCII(rhs.name);
      }
  );
  return JSONSchemaConverter::GenerateObject(sorted_spec, rule_name, need_brace);
}

int32_t GemmaToolCallingConverter::FormatPropertyKey(
    const std::string& key, const SchemaSpecPtr& schema
) {
  XGRAMMAR_CHECK(IsIdentifier(key)) << "A gemma property key must be an identifier: " << key;
  return ByteString(key);
}

std::string GemmaToolCallingConverter::GetKeyPattern() const { return kGemmaVariableName; }

int32_t GemmaToolCallingConverter::BuildBareKeyTrieBody(const BareKeyTrieNode& node, int depth) {
  std::vector<int32_t> choices;
  // A prefix that is not itself a declared name is a complete key on its own.
  if (depth > 0 && !node.is_terminal) {
    choices.push_back(Empty());
  }
  // Any character the trie does not branch on ends the exclusion: the rest is a free identifier.
  choices.push_back(Sequence(
      {builder_.AddCharacterClass(IdentifierRangesExcluding(node.children, depth > 0)),
       builder_.AddCharacterClassStar({{'a', 'z'}, {'A', 'Z'}, {'0', '9'}, {'_', '_'}})}
  ));
  for (const auto& [character, child] : node.children) {
    choices.push_back(Sequence(
        {ByteString(std::string(1, static_cast<char>(character))),
         BuildBareKeyTrieBody(child, depth + 1)}
    ));
  }
  return Choice(choices);
}

int32_t GemmaToolCallingConverter::GetKeyPatternExcluding(
    const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
) {
  if (properties.empty()) {
    return KeyPatternExpression();
  }

  BareKeyTrieNode root;
  for (const auto& property : properties) {
    BareKeyTrieNode* current = &root;
    for (unsigned char character : property.name) {
      current = &current->children[character];
    }
    current->is_terminal = true;
  }

  int32_t key_rule_id = builder_.AddEmptyRuleWithHint(rule_name + "_addl_key");
  builder_.UpdateRuleBody(key_rule_id, BuildBareKeyTrieBody(root, 0));
  return RuleRef(key_rule_id);
}

int32_t GemmaToolCallingConverter::GenerateLiteral(const picojson::value& value) {
  if (value.is<std::string>()) {
    const std::string& text = value.get<std::string>();
    XGRAMMAR_CHECK(
        text.find(kGemmaStringDelim) == std::string::npos &&
        text.find("<|tool_call>") == std::string::npos &&
        text.find("<tool_call|>") == std::string::npos
    ) << "A gemma string literal cannot contain the string delimiter or a tool-call marker";
    return ByteString(kGemmaStringDelim + text + kGemmaStringDelim);
  }
  if (value.is<picojson::object>()) {
    const auto& object = value.get<picojson::object>();
    // The chat template renders mappings with dictsort, whose default is case-insensitive:
    // it compares lowercased keys and keeps insertion order for the ones that tie.
    std::vector<std::string> keys = object.ordered_keys();
    std::stable_sort(keys.begin(), keys.end(), [](const std::string& lhs, const std::string& rhs) {
      return ToLowerASCII(lhs) < ToLowerASCII(rhs);
    });
    std::vector<int32_t> elements;
    elements.push_back(ByteString("{"));
    for (size_t index = 0; index < keys.size(); ++index) {
      if (index != 0) {
        elements.push_back(ByteString(","));
      }
      XGRAMMAR_CHECK(IsIdentifier(keys[index]))
          << "A gemma property key must be an identifier: " << keys[index];
      elements.push_back(ByteString(keys[index] + ":"));
      elements.push_back(GenerateLiteral(object.at(keys[index])));
    }
    elements.push_back(ByteString("}"));
    return Sequence(elements);
  }
  if (value.is<picojson::array>()) {
    const auto& array = value.get<picojson::array>();
    std::vector<int32_t> elements;
    elements.push_back(ByteString("["));
    for (size_t index = 0; index < array.size(); ++index) {
      if (index != 0) {
        elements.push_back(ByteString(","));
      }
      elements.push_back(GenerateLiteral(array[index]));
    }
    elements.push_back(ByteString("]"));
    return Sequence(elements);
  }
  return ByteString(value.serialize());
}

int32_t GemmaToolCallingConverter::GenerateConst(
    const ConstSpec& spec, const std::string& rule_name
) {
  picojson::value value;
  std::string error = picojson::parse(value, spec.json_value);
  XGRAMMAR_CHECK(error.empty()) << "Invalid const JSON value: " << error;
  return GenerateLiteral(value);
}

int32_t GemmaToolCallingConverter::GenerateEnum(
    const EnumSpec& spec, const std::string& rule_name
) {
  XGRAMMAR_DCHECK(!spec.json_values.empty())
      << "GenerateEnum called with empty enum spec for rule: " << rule_name;
  std::vector<int32_t> values;
  values.reserve(spec.json_values.size());
  for (const auto& json_value : spec.json_values) {
    picojson::value value;
    std::string error = picojson::parse(value, json_value);
    XGRAMMAR_CHECK(error.empty()) << "Invalid enum JSON value: " << error;
    values.push_back(GenerateLiteral(value));
  }
  return Choice(values);
}

}  // namespace xgrammar
