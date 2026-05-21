# report_generator.py
# Professional Medical PDF Report - Clean Version
# No emojis in headers - properly formatted

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
    Generate Professional PDF Medical Report
    """
    if not REPORTLAB_OK:
        return None

    # Setup PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch
    )

    # All colors
    DARK_BLUE    = colors.HexColor('#1a237e')
    MED_BLUE     = colors.HexColor('#1565c0')
    LIGHT_BLUE   = colors.HexColor('#e3f2fd')
    SKY_BLUE     = colors.HexColor('#bbdefb')
    RED          = colors.HexColor('#c62828')
    DARK_RED     = colors.HexColor('#b71c1c')
    LIGHT_RED    = colors.HexColor('#ffebee')
    GREEN        = colors.HexColor('#2e7d32')
    DARK_GREEN   = colors.HexColor('#1b5e20')
    LIGHT_GREEN  = colors.HexColor('#e8f5e9')
    ORANGE       = colors.HexColor('#e65100')
    LIGHT_ORANGE = colors.HexColor('#fff3e0')
    PURPLE       = colors.HexColor('#4a148c')
    LIGHT_PURPLE = colors.HexColor('#f3e5f5')
    TEAL         = colors.HexColor('#004d40')
    GRAY         = colors.HexColor('#546e7a')
    LIGHT_GRAY   = colors.HexColor('#f5f5f5')
    WHITE        = colors.white

    # All text styles
    styles = getSampleStyleSheet()

    NORMAL = ParagraphStyle(
        'NORMAL',
        fontSize=10,
        leading=16,
        textColor=colors.HexColor('#333333')
    )
    SMALL = ParagraphStyle(
        'SMALL',
        fontSize=8,
        leading=13,
        textColor=GRAY
    )
    CENTER = ParagraphStyle(
        'CENTER',
        fontSize=9,
        alignment=TA_CENTER,
        textColor=GRAY
    )
    WHITE_BOLD_CENTER = ParagraphStyle(
        'WHITE_BOLD_CENTER',
        fontSize=11,
        alignment=TA_CENTER,
        textColor=WHITE,
        fontName='Helvetica-Bold'
    )
    WHITE_CENTER = ParagraphStyle(
        'WHITE_CENTER',
        fontSize=9,
        alignment=TA_CENTER,
        textColor=WHITE
    )

    # Collect all page elements
    elements = []

    # Get prediction info
    predicted_class = prediction_result.get('predicted_class', 'normal')
    confidence      = prediction_result.get('confidence', 0)
    probabilities   = prediction_result.get('probabilities', {})
    is_cancer       = (predicted_class == 'cancer')

    # Report details
    report_date = datetime.now().strftime("%d %B %Y")
    report_time = datetime.now().strftime("%I:%M %p")
    report_id   = "RPT-" + datetime.now().strftime("%Y%m%d%H%M%S")

    # =========================================================
    # 1. HEADER
    # =========================================================

    # Row 1 - Main title
    row1 = [[Paragraph(
        "AI LUNG CANCER DETECTION SYSTEM",
        ParagraphStyle(
            'TITLE',
            fontSize=22,
            alignment=TA_CENTER,
            textColor=WHITE,
            fontName='Helvetica-Bold',
            leading=26
        )
    )]]
    t1 = Table(row1, colWidths=[7.1 * inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND',     (0, 0), (-1, -1), DARK_BLUE),
        ('TOPPADDING',     (0, 0), (-1, -1), 20),
        ('BOTTOMPADDING',  (0, 0), (-1, -1), 8),
        ('LEFTPADDING',    (0, 0), (-1, -1), 20),
        ('RIGHTPADDING',   (0, 0), (-1, -1), 20),
    ]))
    elements.append(t1)

    # Row 2 - Subtitle
    row2 = [[Paragraph(
        "Medical CT Scan Analysis Report",
        ParagraphStyle(
            'SUBTITLE',
            fontSize=12,
            alignment=TA_CENTER,
            textColor=SKY_BLUE,
            fontName='Helvetica',
            leading=16
        )
    )]]
    t2 = Table(row2, colWidths=[7.1 * inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), colors.HexColor('#283593')),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING',   (0, 0), (-1, -1), 20),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 20),
    ]))
    elements.append(t2)

    # Row 3 - Info bar
    row3 = [[Paragraph(
        "Powered by MobileNetV2 Deep Learning  |  "
        "Transfer Learning  |  For Educational Use Only",
        ParagraphStyle(
            'INFOBAR',
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#90caf9'),
            leading=14
        )
    )]]
    t3 = Table(row3, colWidths=[7.1 * inch])
    t3.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), MED_BLUE),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING',   (0, 0), (-1, -1), 20),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 20),
    ]))
    elements.append(t3)
    elements.append(Spacer(1, 14))

    # =========================================================
    # 2. REPORT AND PATIENT INFORMATION
    # =========================================================

    section_data = [[Paragraph(
        "REPORT AND PATIENT INFORMATION",
        WHITE_BOLD_CENTER
    )]]
    sec = Table(section_data, colWidths=[7.1 * inch])
    sec.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (-1, -1), MED_BLUE),
        ('TOPPADDING',   (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 9),
        ('LEFTPADDING',  (0, 0), (-1, -1), 12),
    ]))
    elements.append(sec)
    elements.append(Spacer(1, 6))

    info_data = [
        [
            Paragraph('<b>Report ID:</b>', NORMAL),
            Paragraph(report_id, NORMAL),
            Paragraph('<b>Date:</b>', NORMAL),
            Paragraph(report_date, NORMAL),
        ],
        [
            Paragraph('<b>Time:</b>', NORMAL),
            Paragraph(report_time, NORMAL),
            Paragraph('<b>Report Type:</b>', NORMAL),
            Paragraph('CT Scan AI Analysis', NORMAL),
        ],
        [
            Paragraph('<b>Patient Name:</b>', NORMAL),
            Paragraph(str(patient_name or 'Not Provided'), NORMAL),
            Paragraph('<b>Age:</b>', NORMAL),
            Paragraph(
                str(patient_age) + " years" if patient_age else "Not Provided",
                NORMAL
            ),
        ],
        [
            Paragraph('<b>Gender:</b>', NORMAL),
            Paragraph(str(patient_gender or 'Not Specified'), NORMAL),
            Paragraph('<b>Scan Type:</b>', NORMAL),
            Paragraph('Chest CT Scan', NORMAL),
        ],
    ]

    info_table = Table(
        info_data,
        colWidths=[1.4 * inch, 2.2 * inch, 1.4 * inch, 2.1 * inch]
    )
    info_table.setStyle(TableStyle([
        ('FONTSIZE',       (0, 0), (-1, -1), 9),
        ('GRID',           (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
        ('TOPPADDING',     (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING',  (0, 0), (-1, -1), 7),
        ('LEFTPADDING',    (0, 0), (-1, -1), 8),
        ('RIGHTPADDING',   (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [LIGHT_GRAY, WHITE]),
        ('BACKGROUND',     (0, 0), (0, -1), LIGHT_BLUE),
        ('BACKGROUND',     (2, 0), (2, -1), LIGHT_BLUE),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 14))

    # =========================================================
    # 3. AI PREDICTION RESULT
    # =========================================================

    section_data2 = [[Paragraph("AI PREDICTION RESULT", WHITE_BOLD_CENTER)]]
    sec2 = Table(section_data2, colWidths=[7.1 * inch])
    sec2.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), MED_BLUE),
        ('TOPPADDING',    (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
        ('LEFTPADDING',   (0, 0), (-1, -1), 12),
    ]))
    elements.append(sec2)
    elements.append(Spacer(1, 6))

    # Result box color
    if is_cancer:
        result_bg   = RED
        result_text = "LUNG CANCER DETECTED"
        result_sub  = (
            "Confidence: " + str(round(confidence, 1)) + "%"
            "   |   Risk Level: HIGH RISK"
            "   |   ACTION REQUIRED IMMEDIATELY"
        )
    else:
        result_bg   = GREEN
        result_text = "NO CANCER DETECTED - NORMAL"
        result_sub  = (
            "Confidence: " + str(round(confidence, 1)) + "%"
            "   |   Risk Level: LOW RISK"
            "   |   Continue Regular Checkups"
        )

    result_box = [
        [Paragraph(result_text, ParagraphStyle(
            'ResultMain',
            fontSize=18,
            alignment=TA_CENTER,
            textColor=WHITE,
            fontName='Helvetica-Bold',
            leading=22
        ))],
        [Paragraph(result_sub, ParagraphStyle(
            'ResultSub',
            fontSize=10,
            alignment=TA_CENTER,
            textColor=WHITE,
            leading=14
        ))]
    ]
    result_table = Table(result_box, colWidths=[7.1 * inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), result_bg),
        ('TOPPADDING',    (0, 0), (0, 0), 16),
        ('BOTTOMPADDING', (0, 0), (0, 0), 6),
        ('TOPPADDING',    (0, 1), (0, 1), 4),
        ('BOTTOMPADDING', (0, 1), (0, 1), 14),
        ('LEFTPADDING',   (0, 0), (-1, -1), 20),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 20),
    ]))
    elements.append(result_table)
    elements.append(Spacer(1, 8))

    # Probability table
    cancer_prob = probabilities.get('cancer', 0)
    normal_prob = probabilities.get('normal', 0)

    prob_data = [
        [
            Paragraph('<b>Cancer Probability</b>', ParagraphStyle(
                'PH', fontSize=10, alignment=TA_CENTER,
                textColor=WHITE, fontName='Helvetica-Bold'
            )),
            Paragraph('<b>Normal Probability</b>', ParagraphStyle(
                'PH2', fontSize=10, alignment=TA_CENTER,
                textColor=WHITE, fontName='Helvetica-Bold'
            )),
            Paragraph('<b>AI Model</b>', ParagraphStyle(
                'PH3', fontSize=10, alignment=TA_CENTER,
                textColor=WHITE, fontName='Helvetica-Bold'
            )),
            Paragraph('<b>Image Input</b>', ParagraphStyle(
                'PH4', fontSize=10, alignment=TA_CENTER,
                textColor=WHITE, fontName='Helvetica-Bold'
            )),
        ],
        [
            Paragraph(str(round(cancer_prob, 1)) + "%", ParagraphStyle(
                'PV1', fontSize=16, alignment=TA_CENTER,
                textColor=RED if is_cancer else DARK_GREEN,
                fontName='Helvetica-Bold'
            )),
            Paragraph(str(round(normal_prob, 1)) + "%", ParagraphStyle(
                'PV2', fontSize=16, alignment=TA_CENTER,
                textColor=DARK_GREEN,
                fontName='Helvetica-Bold'
            )),
            Paragraph('MobileNetV2', ParagraphStyle(
                'PV3', fontSize=10, alignment=TA_CENTER,
                textColor=colors.HexColor('#333333')
            )),
            Paragraph('224 x 224 px', ParagraphStyle(
                'PV4', fontSize=10, alignment=TA_CENTER,
                textColor=colors.HexColor('#333333')
            )),
        ]
    ]

    prob_table = Table(
        prob_data,
        colWidths=[1.8 * inch, 1.8 * inch, 2.0 * inch, 1.5 * inch]
    )
    prob_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), MED_BLUE),
        ('FONTSIZE',      (0, 0), (-1, -1), 10),
        ('GRID',          (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('BACKGROUND',    (0, 1), (0, 1), LIGHT_RED if is_cancer else LIGHT_GREEN),
        ('BACKGROUND',    (1, 1), (1, 1), LIGHT_GREEN),
        ('BACKGROUND',    (2, 1), (-1, 1), WHITE),
    ]))
    elements.append(prob_table)
    elements.append(Spacer(1, 14))

    # =========================================================
    # 4. WHAT THIS MEANS
    # =========================================================

    section3 = [[Paragraph("WHAT THIS RESULT MEANS", WHITE_BOLD_CENTER)]]
    sec3 = Table(section3, colWidths=[7.1 * inch])
    sec3.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), MED_BLUE),
        ('TOPPADDING',    (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
        ('LEFTPADDING',   (0, 0), (-1, -1), 12),
    ]))
    elements.append(sec3)
    elements.append(Spacer(1, 6))

    if is_cancer:
        meaning = (
            "The AI model has detected patterns in your CT scan that are "
            "consistent with lung cancer. The model analyzed over 1,280 "
            "image features using deep learning and found abnormal tissue "
            "patterns with a confidence of "
            + str(round(confidence, 1)) +
            "%. There are suspicious regions in your lung CT scan that "
            "require IMMEDIATE medical attention. This is an AI screening "
            "tool and must be confirmed by a qualified radiologist."
        )
        m_bg     = LIGHT_RED
        m_border = RED
    else:
        meaning = (
            "The AI model has analyzed your CT scan and found NO significant "
            "patterns associated with lung cancer. The model examined over "
            "1,280 image features and determined that your lung tissue "
            "appears normal with "
            + str(round(confidence, 1)) +
            "% confidence. This is a positive result. However, it does not "
            "completely rule out all lung conditions. Regular screenings "
            "are still recommended especially if you smoke or have family "
            "history of lung cancer."
        )
        m_bg     = LIGHT_GREEN
        m_border = GREEN

    meaning_box = [[Paragraph(meaning, NORMAL)]]
    mt = Table(meaning_box, colWidths=[7.1 * inch])
    mt.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), m_bg),
        ('TOPPADDING',    (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING',   (0, 0), (-1, -1), 15),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 15),
        ('LINEBEFORE',    (0, 0), (0, -1), 5, m_border),
    ]))
    elements.append(mt)
    elements.append(Spacer(1, 14))

    # =========================================================
    # 5. IMMEDIATE NEXT STEPS
    # =========================================================

    steps_bg = DARK_RED if is_cancer else MED_BLUE
    section4  = [[Paragraph("IMMEDIATE NEXT STEPS", WHITE_BOLD_CENTER)]]
    sec4 = Table(section4, colWidths=[7.1 * inch])
    sec4.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), steps_bg),
        ('TOPPADDING',    (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
        ('LEFTPADDING',   (0, 0), (-1, -1), 12),
    ]))
    elements.append(sec4)
    elements.append(Spacer(1, 6))

    if is_cancer:
        steps = [
            ("STEP 1 - DO THIS TODAY",
             "Call your nearest cancer hospital or oncology department "
             "IMMEDIATELY and book an urgent appointment. Show them "
             "this report. Do NOT delay this step even by one day."),

            ("STEP 2 - WITHIN 24 TO 48 HOURS",
             "Visit a Pulmonologist (lung specialist) or Oncologist "
             "(cancer specialist). Bring all your previous medical "
             "reports, X-rays, and this AI analysis report with you."),

            ("STEP 3 - GET CONFIRMED DIAGNOSIS",
             "The doctor will order additional tests to confirm the "
             "diagnosis such as PET Scan, Biopsy (tissue sample), "
             "Blood tests for tumor markers, MRI or additional CT scans. "
             "DO NOT self-diagnose based only on this AI report."),

            ("STEP 4 - FOLLOW TREATMENT PLAN",
             "Follow your doctor's treatment plan strictly. Options "
             "may include surgery, chemotherapy, radiation therapy, "
             "or targeted therapy depending on cancer type and stage. "
             "Early treatment gives best results."),

            ("STEP 5 - INFORM YOUR FAMILY",
             "Inform a trusted family member immediately. Having "
             "support during this time is very important for recovery. "
             "Do not face this alone. Family support improves outcomes."),
        ]
        step_bg     = LIGHT_RED
        step_border = RED
    else:
        steps = [
            ("STEP 1 - GOOD NEWS",
             "Your CT scan shows no signs of lung cancer. This is a "
             "very positive result. Continue monitoring your health "
             "regularly and do not ignore any new symptoms that appear."),

            ("STEP 2 - SHARE WITH YOUR DOCTOR",
             "Share this report with your regular doctor or physician "
             "during your next visit. Let them review the AI analysis "
             "result along with your physical examination findings."),

            ("STEP 3 - ANNUAL CT SCREENING",
             "Schedule annual CT screenings especially if you are a "
             "smoker (current or former), aged 50 or above, have a "
             "family history of lung cancer, or work in hazardous "
             "environments like coal mines or chemical factories."),

            ("STEP 4 - HEALTHY LIFESTYLE",
             "Quit smoking immediately if you smoke. Exercise for "
             "30 minutes daily, eat healthy fruits and vegetables, "
             "drink plenty of water, and avoid air pollution. "
             "These steps significantly reduce lung cancer risk."),

            ("STEP 5 - WATCH FOR SYMPTOMS",
             "Consult a doctor immediately if you ever notice: "
             "persistent cough lasting more than 3 weeks, coughing "
             "blood, chest pain, unexplained weight loss, fatigue, "
             "or shortness of breath during normal activities."),
        ]
        step_bg     = LIGHT_GREEN
        step_border = GREEN

    for step_title, step_desc in steps:
        step_box = [[Paragraph(
            "<b>" + step_title + "</b><br/>" + step_desc,
            NORMAL
        )]]
        st = Table(step_box, colWidths=[7.1 * inch])
        st.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), step_bg),
            ('TOPPADDING',    (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('LEFTPADDING',   (0, 0), (-1, -1), 15),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 15),
            ('LINEBEFORE',    (0, 0), (0, -1), 5, step_border),
            ('LINEBELOW',     (0, 0), (-1, -1), 0.5,
             colors.HexColor('#eeeeee')),
        ]))
        elements.append(st)
        elements.append(Spacer(1, 4))

    elements.append(Spacer(1, 10))

    # =========================================================
    # 6. WHICH DOCTOR TO VISIT
    # =========================================================

    section5 = [[Paragraph("WHICH DOCTOR TO VISIT", WHITE_BOLD_CENTER)]]
    sec5 = Table(section5, colWidths=[7.1 * inch])
    sec5.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), PURPLE),
        ('TOPPADDING',    (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
        ('LEFTPADDING',   (0, 0), (-1, -1), 12),
    ]))
    elements.append(sec5)
    elements.append(Spacer(1, 6))

    if is_cancer:
        doctor_rows = [
            ['Specialist', 'What They Do', 'When To Visit'],
            ['Pulmonologist',
             'Lung specialist - first doctor to see',
             'Within 24 hours'],
            ['Oncologist',
             'Cancer specialist - for treatment plan',
             'Within 48 hours'],
            ['Radiologist',
             'Reviews CT scans and imaging',
             'Same as oncologist'],
            ['Pathologist',
             'Analyzes biopsy tissue samples',
             'After biopsy ordered'],
            ['Thoracic Surgeon',
             'Surgery specialist if surgery needed',
             'As recommended'],
        ]
        doc_header_bg = DARK_RED
    else:
        doctor_rows = [
            ['Specialist', 'What They Do', 'When To Visit'],
            ['General Physician',
             'Regular checkup and health monitoring',
             'Every 6 months'],
            ['Pulmonologist',
             'Lung specialist for preventive care',
             'Once a year'],
            ['Radiologist',
             'Annual CT scan reading',
             'Once a year'],
            ['Preventive Medicine',
             'Cancer screening and risk check',
             'Once a year'],
        ]
        doc_header_bg = DARK_GREEN

    doctor_table = Table(
        doctor_rows,
        colWidths=[1.8 * inch, 3.4 * inch, 1.9 * inch]
    )
    doctor_table.setStyle(TableStyle([
        ('BACKGROUND',     (0, 0), (-1, 0), doc_header_bg),
        ('TEXTCOLOR',      (0, 0), (-1, 0), WHITE),
        ('FONTNAME',       (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',       (0, 0), (-1, -1), 9),
        ('GRID',           (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
        ('TOPPADDING',     (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING',  (0, 0), (-1, -1), 8),
        ('LEFTPADDING',    (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_GRAY, WHITE]),
        ('FONTNAME',       (0, 1), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR',      (0, 1), (0, -1), DARK_BLUE),
    ]))
    elements.append(doctor_table)
    elements.append(Spacer(1, 10))

    # Emergency helplines box
    helpline_data = [[Paragraph(
        "<b>EMERGENCY HELPLINES IN INDIA</b><br/><br/>"
        "<b>Cancer Helpline (AIIMS):</b>    1800-11-6117  (Toll Free)<br/>"
        "<b>National Cancer Grid:</b>       1800-209-9626 (Toll Free)<br/>"
        "<b>iCall Mental Health:</b>        9152987821<br/>"
        "<b>Emergency Ambulance:</b>        108<br/>"
        "<b>Health Helpline:</b>            104",
        ParagraphStyle(
            'Helpline',
            fontSize=10,
            leading=20,
            textColor=PURPLE
        )
    )]]
    ht = Table(helpline_data, colWidths=[7.1 * inch])
    ht.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), LIGHT_PURPLE),
        ('TOPPADDING',    (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('LEFTPADDING',   (0, 0), (-1, -1), 16),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 16),
        ('BOX',           (0, 0), (-1, -1), 2, PURPLE),
    ]))
    elements.append(ht)
    elements.append(Spacer(1, 14))


    # 8. DOS AND DONTS
    # =========================================================

    section7 = [[Paragraph("DO'S AND DON'TS", WHITE_BOLD_CENTER)]]
    sec7 = Table(section7, colWidths=[7.1 * inch])
    sec7.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), TEAL),
        ('TOPPADDING',    (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
        ('LEFTPADDING',   (0, 0), (-1, -1), 12),
    ]))
    elements.append(sec7)
    elements.append(Spacer(1, 6))

    if is_cancer:
        dos_list = [
            "Visit doctor IMMEDIATELY today",
            "Bring this report to the hospital",
            "Get a second medical opinion",
            "Tell your family members",
            "Follow doctor's advice strictly",
            "Stay positive - early detection helps",
            "Keep all medical reports organized",
        ]
        donts_list = [
            "Do NOT ignore this result",
            "Do NOT self-medicate at home",
            "Do NOT search symptoms on Google",
            "Do NOT delay your doctor visit",
            "Do NOT smoke or consume alcohol",
            "Do NOT panic - treatment exists",
            "Do NOT rely only on this AI report",
        ]
    else:
        dos_list = [
            "Continue regular health checkups",
            "Get annual CT scan screening done",
            "Exercise 30 minutes every day",
            "Eat fresh fruits and vegetables",
            "Stay hydrated and drink water",
            "Sleep 7 to 8 hours per night",
            "Report any new symptoms to doctor",
        ]
        donts_list = [
            "Do NOT smoke or use tobacco",
            "Do NOT ignore breathing problems",
            "Do NOT skip annual screenings",
            "Do NOT assume you are fully safe",
            "Do NOT ignore persistent cough",
            "Do NOT expose to air pollution",
            "Do NOT rely only on this AI result",
        ]

    dos_text   = "<br/>".join(
        ["<b>YES  -  </b>" + d for d in dos_list]
    )
    donts_text = "<br/>".join(
        ["<b>NO   -  </b>" + d for d in donts_list]
    )

    dos_content = [[
        Paragraph(dos_text, ParagraphStyle(
            'Dos', fontSize=9, leading=18,
            textColor=DARK_GREEN
        )),
        Paragraph(donts_text, ParagraphStyle(
            'Donts', fontSize=9, leading=18,
            textColor=DARK_RED
        )),
    ]]
    dos_table = Table(dos_content, colWidths=[3.5 * inch, 3.6 * inch])
    dos_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (0, 0), LIGHT_GREEN),
        ('BACKGROUND',    (1, 0), (1, 0), LIGHT_RED),
        ('TOPPADDING',    (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING',   (0, 0), (-1, -1), 12),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 12),
        ('GRID',          (0, 0), (-1, -1), 1, WHITE),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(dos_table)
    elements.append(Spacer(1, 14))

    # =========================================================
    # 9. TECHNICAL DETAILS
    # =========================================================

    section8 = [[Paragraph("AI MODEL TECHNICAL DETAILS", WHITE_BOLD_CENTER)]]
    sec8 = Table(section8, colWidths=[7.1 * inch])
    sec8.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), GRAY),
        ('TOPPADDING',    (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
        ('LEFTPADDING',   (0, 0), (-1, -1), 12),
    ]))
    elements.append(sec8)
    elements.append(Spacer(1, 6))

    tech_rows = [
        ['Parameter', 'Value', 'Parameter', 'Value'],
        ['AI Model',      'MobileNetV2',       'Test Accuracy', '99.11%'],
        ['Framework',     'TensorFlow Keras',  'Cancer Recall', '98.8%'],
        ['Image Input',   '224 x 224 pixels',  'Normal Recall', '100%'],
        ['Training Data', '454 CT scan images','Precision',     '97-100%'],
        ['Architecture',  'Transfer Learning', 'F1 Score',      '99%'],
        ['Heatmap',       'Grad-CAM Method',   'Classes',       'Normal / Cancer'],
    ]

    tech_table = Table(
        tech_rows,
        colWidths=[1.5 * inch, 2.1 * inch, 1.5 * inch, 2.0 * inch]
    )
    tech_table.setStyle(TableStyle([
        ('BACKGROUND',     (0, 0), (-1, 0), DARK_BLUE),
        ('TEXTCOLOR',      (0, 0), (-1, 0), WHITE),
        ('FONTNAME',       (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',       (0, 0), (-1, -1), 9),
        ('GRID',           (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
        ('TOPPADDING',     (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING',  (0, 0), (-1, -1), 7),
        ('LEFTPADDING',    (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_GRAY, WHITE]),
        ('FONTNAME',       (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME',       (2, 1), (2, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR',      (0, 1), (0, -1), DARK_BLUE),
        ('TEXTCOLOR',      (2, 1), (2, -1), DARK_BLUE),
    ]))
    elements.append(tech_table)
    elements.append(Spacer(1, 14))

    # =========================================================
    # 10. DISCLAIMER AND FOOTER
    # =========================================================

    elements.append(HRFlowable(
        width="100%", thickness=1,
        color=colors.HexColor('#cccccc'),
        spaceAfter=8
    ))

    disclaimer_box = [
        [Paragraph(
            "IMPORTANT MEDICAL DISCLAIMER",
            ParagraphStyle(
                'DisclTitle',
                fontSize=11,
                fontName='Helvetica-Bold',
                textColor=ORANGE,
                alignment=TA_CENTER
            )
        )],
        [Paragraph(
            "This report is generated by an Artificial Intelligence system "
            "developed as a B.Tech Final Year Academic Project. "
            "It is for EDUCATIONAL AND RESEARCH PURPOSES ONLY. "
            "This AI analysis is NOT a substitute for professional medical "
            "diagnosis, advice, or treatment. The predictions may not be "
            "100% accurate. Always consult a qualified and licensed medical "
            "professional such as a Radiologist or Oncologist for proper "
            "clinical evaluation. The developers are not responsible for "
            "any medical decisions made based on this report.",
            ParagraphStyle(
                'DisclText',
                fontSize=8,
                textColor=GRAY,
                alignment=TA_CENTER,
                leading=14
            )
        )]
    ]
    dt = Table(disclaimer_box, colWidths=[7.1 * inch])
    dt.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), LIGHT_ORANGE),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING',   (0, 0), (-1, -1), 14),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 14),
        ('BOX',           (0, 0), (-1, -1), 1.5, ORANGE),
    ]))
    elements.append(dt)
    elements.append(Spacer(1, 8))

    # Footer line
    elements.append(Paragraph(
        "Generated by AI Lung Cancer Detection System  |  "
        "Report ID: " + report_id + "  |  "
        + report_date + " at " + report_time + "  |  "
        "B.Tech Final Year Project 2024-25",
        ParagraphStyle(
            'FooterLine',
            fontSize=7,
            textColor=GRAY,
            alignment=TA_CENTER
        )
    ))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer