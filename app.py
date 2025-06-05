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
