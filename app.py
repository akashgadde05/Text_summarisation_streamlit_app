from fpdf import FPDF
import base64

def create_pdf(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf_output = bytes(pdf.output(dest='S').encode('latin1'))
    return pdf_output

if result:
    st.success("✅ Summary:")
    summary_text = result[0]
    st.info(summary_text)

    # Create PDF and generate download link
    pdf_data = create_pdf(summary_text)
    b64_pdf = base64.b64encode(pdf_data).decode()

    st.download_button(
        label="📥 Download Summary as PDF",
        data=pdf_data,
        file_name="summary.pdf",
        mime="application/pdf"
    )
    from fpdf import FPDF
import base64

def create_pdf(text, filename="summary.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.output(filename)
    return filename

def download_pdf_button(text, filename="summary.pdf"):
    create_pdf(text, filename)
    with open(filename, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="{filename}">📥 Download Summary as PDF</a>'
    st.markdown(href, unsafe_allow_html=True)

# Show result
if result:
    st.info(result[0])
    download_pdf_button(result[0])

