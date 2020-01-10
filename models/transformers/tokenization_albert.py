"""Tokenization classes."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import collections
import unicodedata
import six
import logging
import sentencepiece as spm

logger = logging.getLogger(__name__)
SPIECE_UNDERLINE = u"▁"

def preprocess_text(inputs,remove_space=True,do_lower_case=True):
  if remove_space:
    outputs = ' '.join(inputs.strip().split())
  else:
    outputs = inputs
  outputs = outputs.replace("``", '"').replace("''", '"')
  if six.PY2 and isinstance(outputs, str):
    outputs = outputs.decode('utf-8')
  outputs = unicodedata.normalize("NFKD", outputs)
  outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
  if do_lower_case:
    outputs = outputs.lower()
  return outputs

def encode_pieces(sp_model, text, return_unicode=True, sample=False):
  """turn sentences into word pieces."""
  text = preprocess_text(text,)
  if six.PY2 and isinstance(text, unicode):
    text = text.encode('utf-8')
  if not sample:
    pieces = sp_model.EncodeAsPieces(text)
  else:
    pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
  new_pieces = []
  for piece in pieces:
    if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
      cur_pieces = sp_model.EncodeAsPieces(
        piece[:-1].replace(SPIECE_UNDERLINE, ''))
      if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
        if len(cur_pieces[0]) == 1:
          cur_pieces = cur_pieces[1:]
        else:
          cur_pieces[0] = cur_pieces[0][1:]
      cur_pieces.append(piece[-1])
      new_pieces.extend(cur_pieces)
    else:
      new_pieces.append(piece)

  # note(zhiliny): convert back to unicode for py2
  if six.PY2 and return_unicode:
    ret_pieces = []
    for piece in new_pieces:
      if isinstance(piece, str):
        piece = piece.decode(piece, "utf-8")
      ret_pieces.append(piece)
    new_pieces = ret_pieces

  return new_pieces

def encode_ids(sp_model, text, sample=False):
  pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
  ids = [sp_model.PieceToId(piece) for piece in pieces]
  return ids


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  with open(vocab_file, "r", encoding="utf-8") as reader:
    tokens = reader.readlines()
  for index, token in enumerate(tokens):
    token = token.rstrip('\n')
    vocab[token] = index
  return vocab

def convert_by_vocab(vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  output = []
  for item in items:
    output.append(vocab[item])
  return output

def convert_tokens_to_ids(vocab, tokens):
  return convert_by_vocab(vocab, tokens)

def convert_ids_to_tokens(inv_vocab, ids):
  return convert_by_vocab(inv_vocab, ids)

def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens

class FullTokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, vocab_file, do_lower_case=True, spm_model_file=None):
    self.vocab = None
    self.sp_model = None
    if spm_model_file:
      self.sp_model = spm.SentencePieceProcessor()
      logger.info("loading sentence piece model")
      self.sp_model.Load(spm_model_file)
      # # Note(mingdachen): For the purpose of consisent API, we are
      # # generating a vocabulary for the sentence piece tokenizer.
      self.vocab = {self.sp_model.IdToPiece(i): i for i
                    in range(self.sp_model.GetPieceSize())}
    else:
      print("load vocab")
      self.vocab = load_vocab(vocab_file)
      print("load token")
      self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
      self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab,unk_token="[UNK]", max_input_chars_per_word=100)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}

  def tokenize(self, text):
    if self.sp_model:
      split_tokens = encode_pieces(self.sp_model, text, return_unicode=False)
    else:
      split_tokens = []
      for token in self.basic_tokenizer.tokenize(text):
        for sub_token in self.wordpiece_tokenizer.tokenize(token):
          split_tokens.append(sub_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    if self.sp_model:
      return [self.sp_model.PieceToId(token) for token in tokens]
    else:
      return convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    if self.sp_model:
      logger.info("using sentence piece tokenzier.")
      return [self.sp_model.IdToPiece(id_) for id_ in ids]
    else:
      return convert_by_vocab(self.inv_vocab, ids)

class BasicTokenizer(object):
  """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

  def __init__(self, do_lower_case=True):
    """Constructs a BasicTokenizer.

    Args:
      do_lower_case: Whether to lower case the input.
    """
    self.do_lower_case = do_lower_case

  def tokenize(self, text):
    """Tokenizes a piece of text."""
    text = self._clean_text(text)

    # This was added on November 1st, 2018 for the multilingual and Chinese
    # models. This is also applied to the English models now, but it doesn't
    # matter since the English models were not trained on any Chinese data
    # and generally don't have any Chinese data in them (there are Chinese
    # characters in the vocabulary because Wikipedia does have some Chinese
    # words in the English Wikipedia.).
    text = self._tokenize_chinese_chars(text)
    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
      if self.do_lower_case:
        token = token.lower()
        token = self._run_strip_accents(token)
      split_tokens.extend(self._run_split_on_punc(token))
    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens

  def _run_strip_accents(self, text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)

  def _run_split_on_punc(self, text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]

  def _tokenize_chinese_chars(self, text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

  def _is_chinese_char(self, cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

  def _clean_text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
      self.vocab = vocab
      self.unk_token = unk_token
      self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
      """Tokenizes a piece of text into its word pieces.

      This uses a greedy longest-match-first algorithm to perform tokenization
      using the given vocabulary.

      For example:
        input = "unaffable"
        output = ["un", "##aff", "##able"]

      Args:
        text: A single token or whitespace separated tokens. This should have
          already been passed through `BasicTokenizer`.

      Returns:
        A list of wordpiece tokens.
      """

      output_tokens = []
      for token in whitespace_tokenize(text):
        chars = list(token)
        if len(chars) > self.max_input_chars_per_word:
          output_tokens.append(self.unk_token)
          continue

        is_bad = False
        start = 0
        sub_tokens = []
        while start < len(chars):
          end = len(chars)
          cur_substr = None
          while start < end:
            substr = "".join(chars[start:end])
            if start > 0:
              substr = "##" + substr
            if substr in self.vocab:
              cur_substr = substr
              break
            end -= 1
          if cur_substr is None:
            is_bad = True
            break
          sub_tokens.append(cur_substr)
          start = end

        if is_bad:
          output_tokens.append(self.unk_token)
        else:
          output_tokens.extend(sub_tokens)
      return output_tokens

def _is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  # \t, \n, and \r are technically control characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def _is_control(char):
  """Checks whether `chars` is a control character."""
  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat in ("Cc", "Cf"):
    return True
  return False

def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False
