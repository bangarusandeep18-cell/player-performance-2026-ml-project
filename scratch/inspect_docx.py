import docx

doc = docx.Document('ML_Project_Report_Template (1).docx')
for i, p in enumerate(doc.paragraphs):
    if 'rId6' in p._p.xml or 'rId7' in p._p.xml:
        print(f"Paragraph {i}: text='{p.text}'")
        for r in p.runs:
            print(f"   Run: text='{r.text}'")
