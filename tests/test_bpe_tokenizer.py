import os
import shutil
import tempfile
import unittest
from src.data.bpe_tokenizer import (
    ByteLevelBPETokenizer,
    bytes_to_unicode,
    unicode_to_bytes,
)


class TestByteLevelBPETokenizer(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_bytes_to_unicode_bijection(self):
        """Verify that all 256 bytes map uniquely to unicode and invert losslessly."""
        b2u = bytes_to_unicode()
        u2b = unicode_to_bytes()

        self.assertEqual(len(b2u), 256)
        self.assertEqual(len(u2b), 256)

        for b in range(256):
            char = b2u[b]
            self.assertEqual(u2b[char], b)

    def test_base_vocab_initialization(self):
        """Verify initial vocabulary contains special tokens + 256 byte tokens."""
        tok = ByteLevelBPETokenizer()
        self.assertEqual(len(tok), 260)  # 4 special + 256 byte tokens
        self.assertEqual(tok.unk_id, 0)
        self.assertEqual(tok.pad_id, 1)
        self.assertEqual(tok.bos_id, 2)
        self.assertEqual(tok.eos_id, 3)

    def test_training_and_roundtrip(self):
        """Test training on a small corpus and lossless encode/decode roundtrip."""
        corpus = [
            "Eine schöne junge Frau läuft über die grüne Wiese.",
            "Ein kleiner brauner Hund spielt mit einem Ball im Garten.",
            "Die Katze schläft auf dem gemütlichen Sofa.",
            "Zwei Kinder essen frische Äpfel und Erdbeeren am Flussufer.",
            "Hallo Welt! Wie geht es Ihnen heute? 12345 € & %.",
        ]

        tok = ByteLevelBPETokenizer()
        tok.train(corpus, vocab_size=320, min_frequency=1)

        self.assertGreater(len(tok), 260)
        self.assertLessEqual(len(tok), 320)

        for sentence in corpus:
            encoded_ids = tok.encode(sentence, add_special_tokens=True)
            self.assertEqual(encoded_ids[0], tok.bos_id)
            self.assertEqual(encoded_ids[-1], tok.eos_id)

            decoded_text = tok.decode(encoded_ids, skip_special_tokens=True)
            self.assertEqual(sentence, decoded_text)

    def test_german_special_characters_lossless(self):
        """Verify German umlauts and symbols decode losslessly even if unseen during training."""
        corpus = ["Das ist ein einfacher Satz zum Lernen."]
        tok = ByteLevelBPETokenizer()
        tok.train(corpus, vocab_size=280, min_frequency=1)

        unseen_german = "Große Vögel fliegen über die Städte und Häuser: ä ö ü ß Ä Ö Ü €."
        ids = tok.encode(unseen_german, add_special_tokens=False)
        recovered = tok.decode(ids, skip_special_tokens=True)
        self.assertEqual(unseen_german, recovered)

    def test_save_and_load(self):
        """Test saving tokenizer to JSON and loading it back."""
        corpus = ["Trainiere den Tokenizer mit ein paar deutschen Beispielen."]
        tok = ByteLevelBPETokenizer()
        tok.train(corpus, vocab_size=290, min_frequency=1)

        save_path = os.path.join(self.test_dir, "test_tok.json")
        tok.save(save_path)

        loaded_tok = ByteLevelBPETokenizer.load(save_path)
        self.assertEqual(len(tok), len(loaded_tok))
        self.assertEqual(tok.vocab, loaded_tok.vocab)
        self.assertEqual(tok.merges, loaded_tok.merges)

        test_sent = "Ein neuer Test-Satz mit Umlauten: Überprüfung."
        self.assertEqual(
            tok.encode(test_sent),
            loaded_tok.encode(test_sent),
        )
        self.assertEqual(
            tok.decode(tok.encode(test_sent)),
            loaded_tok.decode(loaded_tok.encode(test_sent)),
        )


if __name__ == "__main__":
    unittest.main()
