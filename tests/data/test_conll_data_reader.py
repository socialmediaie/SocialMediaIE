from io import StringIO
from unittest import TestCase
from SocialMediaIE.data.conll_data_reader import read_conll_file

FILE_CONTENT = """Hello    NN
World   VBP
Line    ADV

Hello   NN
New VBP
World   ADV

Hello   NN
This    CONJ
World   ADV
"""


class TestConllDataReader(TestCase):
    def setUp(self):
        self.fp = StringIO(FILE_CONTENT)
    
    def test_read_conll_file(self):
        expected_conll_lines = [
            ['Hello    NN', 'World   VBP', 'Line    ADV'],
            ['Hello   NN', 'New VBP', 'World   ADV'],
            ['Hello   NN', 'This    CONJ', 'World   ADV']
        ]
        extracted_lines = list(read_conll_file(self.fp))
        self.assertEqual(extracted_lines, expected_conll_lines)