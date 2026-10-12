import docx

doc = docx.Document('ML_Project_Report_Template (1).docx')
for style_name in ['Normal', 'Heading 1', 'Heading 2']:
    s = doc.styles[style_name]
    print(f"Style: {style_name}")
    if s.font:
        print(f"  font name: {s.font.name}, size: {s.font.size.pt if s.font.size else 'None'}, bold: {s.font.bold}")
