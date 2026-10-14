"""Small original study notes; no third-party textbooks or external fixtures."""
from io import BytesIO
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

SAMPLE_PAGES = [
    ["Sample science notes: Electricity", "Original practice material for testing NCERT Tutor, not an official textbook.", "Electric current is the rate of flow of electric charge. I = Q / t.", "The SI unit of current is the ampere. An ammeter is connected in series.", "Potential difference is work done per unit charge. V = W / Q.", "The SI unit of potential difference is the volt.", "Ohm's law: at constant temperature, voltage equals current times resistance.", "V = I R. Resistance is measured in ohms.", "Example: 6 volts across 3 ohms produces a current of 2 amperes."],
    ["Sample science notes: Circuits and power", "In a series circuit, the same current flows through every resistor.", "Total series resistance is R = R1 + R2 + R3.", "In a parallel circuit, each branch has the same potential difference.", "For parallel resistors, 1/R = 1/R1 + 1/R2 + 1/R3.", "Electric power is the rate of transfer of electrical energy.", "P = V I. Using Ohm's law, P = I squared R, and P = V squared / R.", "The SI unit of power is the watt. Energy transferred is E = P t.", "Example: a 60 watt lamp used for 2 hours consumes 120 watt-hours.", "A fuse protects a circuit by melting when the current is too large."]
]

def make_pdf(pages=None):
    writer = PdfWriter()
    for lines in pages or SAMPLE_PAGES:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        escaped = [line.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)') for line in lines]
        stream = DecodedStreamObject()
        stream.set_data(('BT /F1 11 Tf 48 736 Td 24 TL ' + ' T* '.join(f'({line}) Tj' for line in escaped) + ' ET').encode('ascii'))
        page[NameObject('/Contents')] = writer._add_object(stream)
    output = BytesIO(); writer.write(output)
    return output.getvalue()
