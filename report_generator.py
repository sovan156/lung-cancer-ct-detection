# report_generator.py
# Generates downloadable PDF medical reports

import io
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer,
        Table, TableStyle, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False


def check_reportlab():
    return REPORTLAB_OK


def generate_pdf_report(patient_name, patient_age,
                        patient_gender, prediction_result):
    """
    Generate a professional PDF medical report

    Args:
        patient_name      : patient name string
        patient_age       : patient age number
        patient_gender    : 'Male' or 'Female'
        prediction_result : result dict from predict.py

    Returns:
        PDF as bytes (for download button)
        None if reportlab not installed
    """
    if not REPORTLAB_OK:
        return None

    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.75*inch,   bottomMargin=0.75*inch
    )

    styles   = getSampleStyleSheet()
    elements = []

    # Title
    title_style = ParagraphStyle(
        'Title', parent=styles['Title'],
        fontSize=20, alignment=TA_CENTER,
        textColor=colors.HexColor('#1a237e'),
        fontName='Helvetica-Bold', spaceAfter=4
    )
    sub_style = ParagraphStyle(
        'Sub', parent=styles['Normal'],
        fontSize=10, alignment=TA_CENTER,
        textColor=colors.HexColor('#546e7a')
    )
    header_style = ParagraphStyle(
        'Header', parent=styles['Heading1'],
        fontSize=13, textColor=colors.HexColor('#1565c0'),
        fontName='Helvetica-Bold',
        spaceBefore=12, spaceAfter=6
    )
    normal_style = ParagraphStyle(
        'Normal2', parent=styles['Normal'],
        fontSize=10, leading=16,
        textColor=colors.HexColor('#333333')
    )

    # ── Hospital Header ────────────────────────────────────
    elements.append(Paragraph(
        "AI-Powered Lung Cancer Detection System",
        title_style
    ))
    elements.append(Paragraph(
        "CT Scan Analysis Medical Report",
        sub_style
    ))
    elements.append(HRFlowable(
        width="100%", thickness=2,
        color=colors.HexColor('#1565c0'),
        spaceAfter=12
    ))

    # ── Report Details ─────────────────────────────────────
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    report_id   = f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    details_data = [
        ['Report ID:', report_id,   'Date:', report_date],
        ['Analysis:',  'CT Scan',   'Model:', 'MobileNetV2 Deep Learning']
    ]
    details_table = Table(
        details_data,
        colWidths=[1.2*inch, 2.5*inch, 1.0*inch, 2.6*inch]
    )
    details_table.setStyle(TableStyle([
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('FONTNAME',   (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',   (2,0), (2,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',  (0,0), (0,-1), colors.HexColor('#1565c0')),
        ('TEXTCOLOR',  (2,0), (2,-1), colors.HexColor('#1565c0')),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f5f5f5')),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
        ('PADDING',    (0,0), (-1,-1), 6),
    ]))
    elements.append(details_table)
    elements.append(Spacer(1, 12))

    # ── Patient Info ───────────────────────────────────────
    elements.append(Paragraph("Patient Information", header_style))

    patient_data = [
        ['Name:',   patient_name or 'Not Provided',
         'Age:',    f"{patient_age} years" if patient_age else 'N/A'],
        ['Gender:', patient_gender or 'Not Specified',
         'Test:',   'Lung CT Scan Analysis']
    ]
    patient_table = Table(
        patient_data,
        colWidths=[1.2*inch, 2.5*inch, 1.0*inch, 2.6*inch]
    )
    patient_table.setStyle(TableStyle([
        ('FONTSIZE',   (0,0), (-1,-1), 10),
        ('FONTNAME',   (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',   (2,0), (2,-1), 'Helvetica-Bold'),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#e3f2fd')),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor('#e3f2fd')),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#90caf9')),
        ('PADDING',    (0,0), (-1,-1), 8),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 15))

    # ── AI Result ──────────────────────────────────────────
    elements.append(Paragraph("AI Analysis Result", header_style))

    predicted_class = prediction_result.get('predicted_class', 'unknown')
    confidence      = prediction_result.get('confidence', 0)
    probabilities   = prediction_result.get('probabilities', {})

    if predicted_class == 'malignant':
        res_color = colors.HexColor('#f44336')
        res_bg    = colors.HexColor('#ffebee')
        res_text  = 'MALIGNANT - CANCER DETECTED'
    elif predicted_class == 'benign':
        res_color = colors.HexColor('#ff9800')
        res_bg    = colors.HexColor('#fff3e0')
        res_text  = 'BENIGN TUMOR DETECTED'
    else:
        res_color = colors.HexColor('#4caf50')
        res_bg    = colors.HexColor('#e8f5e9')
        res_text  = 'NORMAL - NO CANCER DETECTED'

    result_data = [
        [f'PREDICTION: {res_text}'],
        [f'AI Confidence: {confidence:.1f}%']
    ]
    result_table = Table(result_data, colWidths=[7.3*inch])
    result_table.setStyle(TableStyle([
        ('FONTNAME',      (0,0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (0,0), 14),
        ('FONTSIZE',      (0,1), (0,1), 11),
        ('TEXTCOLOR',     (0,0), (-1,-1), res_color),
        ('BACKGROUND',    (0,0), (-1,-1), res_bg),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING',    (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('BOX',           (0,0), (-1,-1), 2, res_color),
    ]))
    elements.append(result_table)
    elements.append(Spacer(1, 12))

    # Probabilities table
    prob_data = [['Class', 'Probability', 'Risk Level']]
    for cls, prob in probabilities.items():
        risk = ('HIGH' if cls == 'malignant'
                else 'MEDIUM' if cls == 'benign' else 'LOW')
        prob_data.append([cls.upper(), f'{prob:.1f}%', risk])

    prob_table = Table(
        prob_data,
        colWidths=[2.5*inch, 2.5*inch, 2.3*inch]
    )
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1565c0')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 10),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
        ('PADDING',    (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#f8f9fa'), colors.white])
    ]))
    elements.append(prob_table)
    elements.append(Spacer(1, 15))

    # ── Recommendation ─────────────────────────────────────
    elements.append(Paragraph("Clinical Recommendation", header_style))
    description = prediction_result.get('description', '')
    elements.append(Paragraph(description, normal_style))
    elements.append(Spacer(1, 20))

    # ── Disclaimer ─────────────────────────────────────────
    elements.append(HRFlowable(
        width="100%", thickness=1,
        color=colors.HexColor('#cccccc'), spaceAfter=8
    ))
    elements.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI system for "
        "educational and research purposes only. It is NOT a substitute "
        "for professional medical diagnosis. Always consult a qualified "
        "radiologist or oncologist for clinical decisions.",
        ParagraphStyle('Disc', parent=styles['Normal'],
                       fontSize=8, alignment=TA_CENTER,
                       textColor=colors.HexColor('#888888'))
    ))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer
