# -*- coding: utf-8 -*-
# Standard library imports
import os
from datetime import datetime
import re

# Third-party imports
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Frame, PageTemplate, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus.paragraph import ParaLines

# Define RGB values for custom colors - using more subtle blues
DARK_BLUE_RGB = (44/255, 62/255, 80/255)      # More muted dark blue
MEDIUM_BLUE_RGB = (52/255, 73/255, 94/255)    # More muted medium blue

class PDFReportGenerator:
    def __init__(self, output_folder, llm_name, project_name):
        self.output_folder = output_folder
        self.llm_name = llm_name
        self.project_name = project_name
        self.report_title = None
        self.styles = self._create_styles()
        self.debug_mode = False  # Set to True to print debug info

    def create_enhanced_pdf_report(self, findings, pdf_content, image_data, filename="report", report_title=None):
        self.report_title = report_title or f"Analysis Report for {self.project_name}"
        pdf_file = os.path.join(self.output_folder, f"{filename}.pdf")
        doc = SimpleDocTemplate(pdf_file, pagesize=A4)

        elements = []

        # Cover page
        elements.extend(self._create_cover_page(doc))

        # Table of Contents
        elements.append(Paragraph("Table of Contents", self.styles['Heading1']))
        
        # Add key findings entry
        elements.append(Paragraph("Key Findings", self.styles['TOCEntry']))
        elements.append(Spacer(1, 6))
        
        # Add content entries
        for i, (analysis_type, _, _) in enumerate(pdf_content):
            elements.append(Paragraph(f"{i+1}. {analysis_type}", self.styles['TOCEntry']))
            elements.append(Spacer(1, 4))  # Small space between entries
        
        elements.append(PageBreak())

        # Key Findings
        if findings:
            elements.append(Paragraph("Key Findings", self.styles['Heading1']))
            for finding in findings:
                elements.extend(self._text_to_reportlab(finding))
            elements.append(PageBreak())

        # Main content
        for i, (analysis_type, image_paths, interpretation) in enumerate(pdf_content):
            # Add section numbering with smaller heading
            elements.append(Paragraph(f"{i+1}. {analysis_type}", self.styles['ControlTitle']))
            
            # Process interpretation text, ensuring we don't re-add the title
            processed_content = self._text_to_reportlab(interpretation, skip_title=True, control_title=analysis_type)
            elements.extend(processed_content)

            # Add images for this analysis type
            for description, img_path in image_paths:
                if os.path.exists(img_path):
                    img = Image(img_path)
                    available_width = doc.width
                    aspect = img.drawHeight / img.drawWidth
                    img.drawWidth = available_width
                    img.drawHeight = available_width * aspect
                    elements.append(img)
                    elements.append(Paragraph(description, self.styles['Caption']))
                    elements.append(Spacer(1, 12))

            elements.append(PageBreak())

        try:
            doc.build(elements, onFirstPage=self._add_header_footer, onLaterPages=self._add_header_footer)
            print(f"PDF report saved to {pdf_file}")
            return pdf_file
        except Exception as e:
            print(f"Error building PDF: {str(e)}")
            return None

    def _create_styles(self):
        styles = getSampleStyleSheet()
        
        # Modify existing styles
        styles['Title'].fontSize = 24
        styles['Title'].alignment = TA_CENTER
        styles['Title'].textColor = colors.white
        styles['Title'].backColor = colors.Color(*MEDIUM_BLUE_RGB)
        styles['Title'].spaceAfter = 12
        styles['Title'].spaceBefore = 12
        styles['Title'].leading = 30

        # Reduce size and make color more subtle for all headings
        styles['Heading1'].fontSize = 14  # Reduced from 18
        styles['Heading1'].alignment = TA_JUSTIFY
        styles['Heading1'].spaceAfter = 10
        styles['Heading1'].spaceBefore = 8
        styles['Heading1'].textColor = colors.Color(*MEDIUM_BLUE_RGB)

        # Add a custom style for control titles (smaller than Heading1)
        styles.add(ParagraphStyle(
            name='ControlTitle',
            parent=styles['Heading1'],
            fontSize=12,  # Smaller than standard headings
            textColor=colors.Color(*DARK_BLUE_RGB),
            spaceBefore=8,
            spaceAfter=8,
            fontName='Helvetica-Bold'
        ))

        # Modify or add Heading2 style
        if 'Heading2' in styles:
            styles['Heading2'].fontSize = 12  # Reduced from 16
            styles['Heading2'].textColor = colors.Color(*DARK_BLUE_RGB)
            styles['Heading2'].spaceBefore = 6
            styles['Heading2'].spaceAfter = 4
        else:
            styles.add(ParagraphStyle(
                name='Heading2',
                parent=styles['Heading1'],
                fontSize=12,
                textColor=colors.Color(*DARK_BLUE_RGB),
                spaceBefore=6,
                spaceAfter=4
            ))
        
        # Modify or add Heading3 style
        if 'Heading3' in styles:
            styles['Heading3'].fontSize = 11  # Reduced from 14
            styles['Heading3'].textColor = colors.black
            styles['Heading3'].spaceBefore = 5
            styles['Heading3'].spaceAfter = 3
        else:
            styles.add(ParagraphStyle(
                name='Heading3',
                parent=styles['Heading2'],
                fontSize=11,
                textColor=colors.black,
                spaceBefore=5,
                spaceAfter=3
            ))

        # Modify or add Heading4 style
        if 'Heading4' in styles:
            styles['Heading4'].fontSize = 10  # Reduced from 12
            styles['Heading4'].fontName = 'Helvetica-Bold'
            styles['Heading4'].textColor = colors.black
            styles['Heading4'].spaceBefore = 3
            styles['Heading4'].spaceAfter = 2
            styles['Heading4'].leading = 12
        else:
            styles.add(ParagraphStyle(
                name='Heading4',
                parent=styles['Heading3'],
                fontSize=10,
                fontName='Helvetica-Bold',
                textColor=colors.black,
                spaceBefore=3,
                spaceAfter=2,
                leading=12
            ))

        # Add a style for expert comments section heading
        styles.add(ParagraphStyle(
            name='ExpertCommentsHeading',
            parent=styles['Normal'],
            fontSize=10,
            fontName='Helvetica-Bold',
            textColor=colors.Color(*DARK_BLUE_RGB),
            spaceBefore=6,
            spaceAfter=2,
            leading=12
        ))

        # Modify Normal style
        styles['Normal'].fontSize = 10
        styles['Normal'].alignment = TA_JUSTIFY
        styles['Normal'].spaceAfter = 6
        styles['Normal'].textColor = colors.black
        styles['Normal'].leading = 14  # Better line spacing

        # Add or modify BulletPoint style
        if 'BulletPoint' in styles:
            styles['BulletPoint'].bulletIndent = 20
            styles['BulletPoint'].leftIndent = 40
            styles['BulletPoint'].firstLineIndent = -20
            styles['BulletPoint'].spaceBefore = 2
            styles['BulletPoint'].spaceAfter = 2
        else:
            styles.add(ParagraphStyle(
                name='BulletPoint',
                parent=styles['Normal'],
                bulletIndent=20,
                leftIndent=40,
                firstLineIndent=-20,
                spaceBefore=2,
                spaceAfter=2
            ))
        
        # Add or modify SubBulletPoint style
        if 'SubBulletPoint' in styles:
            styles['SubBulletPoint'].bulletIndent = 40
            styles['SubBulletPoint'].leftIndent = 60
            styles['SubBulletPoint'].firstLineIndent = -20
            styles['SubBulletPoint'].spaceBefore = 1
            styles['SubBulletPoint'].spaceAfter = 1
        else:
            styles.add(ParagraphStyle(
                name='SubBulletPoint',
                parent=styles['BulletPoint'],
                bulletIndent=40,
                leftIndent=60,
                firstLineIndent=-20,
                spaceBefore=1,
                spaceAfter=1
            ))
            
        # Add style for bullet points with embedded bold text
        styles.add(ParagraphStyle(
            name='BulletWithEmbeddedBold',
            parent=styles['BulletPoint'],
            bulletIndent=20,
            leftIndent=40,
            firstLineIndent=-20,
            spaceBefore=2,
            spaceAfter=2
        ))

        # Add or modify TOCEntry style
        if 'TOCEntry' in styles:
            styles['TOCEntry'].fontSize = 11
            styles['TOCEntry'].leftIndent = 20
            styles['TOCEntry'].firstLineIndent = -20
            styles['TOCEntry'].spaceBefore = 2
            styles['TOCEntry'].spaceAfter = 2
        else:
            styles.add(ParagraphStyle(
                name='TOCEntry',
                parent=styles['Normal'],
                fontSize=11,
                leftIndent=20,
                firstLineIndent=-20,
                spaceBefore=2,
                spaceAfter=2
            ))

        # Add or modify Caption style
        if 'Caption' in styles:
            styles['Caption'].fontSize = 8
            styles['Caption'].alignment = TA_CENTER
            styles['Caption'].spaceAfter = 6
            styles['Caption'].textColor = colors.Color(*DARK_BLUE_RGB)
            styles['Caption'].fontName = 'Helvetica-Bold'
        else:
            styles.add(ParagraphStyle(
                name='Caption',
                parent=styles['Normal'],
                fontSize=8,
                alignment=TA_CENTER,
                spaceAfter=6,
                textColor=colors.Color(*DARK_BLUE_RGB),
                fontName='Helvetica-Bold'
            ))
            
        # Add or modify SectionTitle style
        if 'SectionTitle' in styles:
            styles['SectionTitle'].fontSize = 11
            styles['SectionTitle'].fontName = 'Helvetica-Bold'
            styles['SectionTitle'].spaceBefore = 6
            styles['SectionTitle'].spaceAfter = 2
        else:
            styles.add(ParagraphStyle(
                name='SectionTitle',
                parent=styles['Normal'],
                fontSize=11,
                fontName='Helvetica-Bold',
                spaceBefore=6,
                spaceAfter=2
            ))
            
        return styles

    def _create_cover_page(self, doc):
        def draw_background(canvas, doc):
            canvas.saveState()
            canvas.setFillColor(colors.Color(*MEDIUM_BLUE_RGB))
            canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=1)
            canvas.restoreState()

        cover_frame = Frame(
            doc.leftMargin, 
            doc.bottomMargin, 
            doc.width, 
            doc.height,
            id='CoverFrame'
        )
        cover_template = PageTemplate(id='CoverPage', frames=[cover_frame], onPage=draw_background)
        doc.addPageTemplates([cover_template])

        elements = []
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph(self.report_title, self.styles['Title']))
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d')}", self.styles['Normal']))
        elements.append(Paragraph(f"AI-powered analysis by ERAG using {self.llm_name}", self.styles['Normal']))
        elements.append(Spacer(1, 0.25*inch))
        elements.append(Paragraph("Includes maturity-based implementation approaches and expert guidance", self.styles['Normal']))
        elements.append(PageBreak())

        # Add a normal template for subsequent pages
        normal_frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='NormalFrame')
        normal_template = PageTemplate(id='NormalPage', frames=[normal_frame], onPage=self._add_header_footer)
        doc.addPageTemplates([normal_template])

        return elements

    def _add_header_footer(self, canvas, doc):
        canvas.saveState()
        
        # Header
        canvas.setFillColor(colors.Color(*DARK_BLUE_RGB))
        canvas.setFont('Helvetica-Bold', 8)
        canvas.drawString(inch, doc.pagesize[1] - 0.5*inch, self.report_title)
        
        # Footer
        canvas.setFillColor(colors.Color(*DARK_BLUE_RGB))
        canvas.setFont('Helvetica', 8)
        canvas.drawString(inch, 0.5 * inch, f"Page {doc.page}")
        canvas.drawRightString(doc.pagesize[0] - inch, 0.5 * inch, "Powered by ERAG")

        canvas.restoreState()

    def _text_to_reportlab(self, text, skip_title=False, control_title=None):
        """Convert text with Markdown-like formatting to ReportLab elements with improved formatting."""
        elements = []
        paragraphs = []
        current_paragraph = []
        
        # Split the text into individual lines
        lines = text.split('\n')
        
        in_list = False
        list_items = []
        list_level = 0
        skipped_title = False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Skip the first heading if requested (to avoid title duplication)
            if skip_title and not skipped_title and (
                line.strip().startswith('# ') or 
                line.strip().startswith('## ') or 
                line.strip().startswith('### ') or 
                (control_title and line.strip() == control_title or
                 control_title and control_title in line)
            ):
                skipped_title = True
                i += 1
                # Skip any empty lines after the title too
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i >= len(lines):
                    break
                line = lines[i]
            
            # Skip empty lines at the beginning
            if not line.strip() and not current_paragraph:
                i += 1
                continue
            
            # Handle expert comments heading
            if line.strip() == "**Expert Comments:**":
                # If there's a current paragraph, add it
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                
                # Add the expert comments heading
                paragraphs.append(('expert_comments_heading', line.strip()))
                i += 1
                continue
                
            # Handle Markdown headings with new smaller styles
            if line.startswith('# '):
                # If there's a current paragraph, add it to the paragraphs list
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                
                # Add the heading
                paragraphs.append(('heading1', line[2:]))
                
                # Add extra spacing after headings
                paragraphs.append('')
                
            elif line.startswith('## '):
                # If there's a current paragraph, add it to the paragraphs list
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                
                # Add the heading
                paragraphs.append(('heading2', line[3:]))
                
                # Add extra spacing after headings
                paragraphs.append('')
                
            elif line.startswith('### '):
                # If there's a current paragraph, add it to the paragraphs list
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                
                # Add the heading
                paragraphs.append(('heading3', line[4:]))
                
                # Add extra spacing after headings
                paragraphs.append('')
                
            elif line.startswith('#### '):
                # If there's a current paragraph, add it to the paragraphs list
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                
                # Add the heading
                paragraphs.append(('heading4', line[5:]))
                
                # Add extra spacing after headings
                paragraphs.append('')
            
            # Handle list items
            elif line.strip().startswith('* ') or line.strip().startswith('- '):
                # If there's a current paragraph, add it
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                
                # Determine list level based on indentation
                indent = len(line) - len(line.lstrip())
                if indent >= 4:
                    list_level = 1  # Sub-bullet
                else:
                    list_level = 0  # Main bullet
                
                # Remove the bullet point symbol
                item_text = line.strip()[2:].strip()
                
                # Start/continue a list
                if not in_list:
                    # If we're starting a new list, add extra space before it
                    if paragraphs and paragraphs[-1] != '':
                        paragraphs.append('')
                    in_list = True
                    list_items = []
                
                # Look for multi-line bullet points - collect all lines until next bullet or blank line
                bullet_content = [item_text]
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith('* ') and not lines[j].strip().startswith('- ') and lines[j].strip():
                    bullet_content.append(lines[j].strip())
                    j += 1
                
                # Process multi-lines together for the bullet point
                full_bullet_content = ' '.join(bullet_content)
                
                # Add this item to the list with formatting info
                list_items.append((list_level, full_bullet_content))
                
                # Skip the lines we've already processed
                i = j - 1
                
                # Check if the next line is also a list item
                if i + 1 < len(lines) and (lines[i+1].strip().startswith('* ') or lines[i+1].strip().startswith('- ')):
                    # If the next line is also a list item, continue
                    i += 1
                    continue
                else:
                    # End of list, add it as a special item
                    paragraphs.append(('bullet_list', list_items))
                    in_list = False
            
            # Handle empty lines as paragraph breaks
            elif not line.strip():
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                
                # Only add empty paragraph if the last one wasn't already empty
                if paragraphs and paragraphs[-1] != '':
                    paragraphs.append('')
            
            # Handle bold section titles (e.g., "**Title:**")
            elif line.strip().startswith('**') and '**:' in line:
                # If there's a current paragraph, add it
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                
                # Add as a section title
                paragraphs.append(('section_title', line.strip()))
            
            # Regular text - add to the current paragraph
            else:
                current_paragraph.append(line)
            
            i += 1
        
        # Add any remaining paragraph
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Convert paragraphs to ReportLab elements
        for p in paragraphs:
            if not p:
                # Empty paragraph - add a spacer
                elements.append(Spacer(1, 6))
            elif isinstance(p, tuple):
                if p[0] == 'heading1':
                    elements.append(Paragraph(p[1], self.styles['Heading1']))
                    elements.append(Spacer(1, 6))
                elif p[0] == 'heading2':
                    elements.append(Paragraph(p[1], self.styles['Heading2']))
                    elements.append(Spacer(1, 4))
                elif p[0] == 'heading3':
                    elements.append(Paragraph(p[1], self.styles['Heading3']))
                    elements.append(Spacer(1, 3))
                elif p[0] == 'heading4':
                    elements.append(Paragraph(p[1], self.styles['Heading4']))
                    elements.append(Spacer(1, 2))
                elif p[0] == 'expert_comments_heading':
                    # Add expert comments heading with a special style
                    elements.append(Paragraph(p[1], self.styles['ExpertCommentsHeading']))
                    elements.append(Spacer(1, 2))
                elif p[0] == 'section_title':
                    # Clean up the section title formatting
                    title = p[1].replace('**', '')
                    elements.append(Paragraph(title, self.styles['SectionTitle']))
                elif p[0] == 'bullet_list':
                    # Add each bullet item as a paragraph with proper indentation
                    for level, content in p[1]:
                        # Process markdown formatting in the bullet content
                        formatted_content = self._process_bullet_content(content)
                        
                        if level == 0:
                            # Main bullet
                            bullet_text = f"• {formatted_content}"
                            elements.append(Paragraph(bullet_text, self.styles['BulletWithEmbeddedBold']))
                        else:
                            # Sub-bullet
                            bullet_text = f"   ○ {formatted_content}"
                            elements.append(Paragraph(bullet_text, self.styles['BulletWithEmbeddedBold']))
                    
                    elements.append(Spacer(1, 3))  # Add space after list
            else:
                try:
                    # Process any markdown in the paragraph
                    processed_text = self._process_markdown_inline(p)
                    
                    # Create a paragraph with the processed text
                    para = Paragraph(processed_text, self.styles['Normal'])
                    elements.append(para)
                except Exception as e:
                    # If there's an error, try with a cleaned version
                    cleaned_text = self._clean_text(p)
                    try:
                        para = Paragraph(cleaned_text, self.styles['Normal'])
                        elements.append(para)
                    except:
                        # If it still fails, add the text as a simple string
                        print(f"Warning: Could not parse paragraph: {cleaned_text[:50]}...")
                        elements.append(cleaned_text)
        
        return elements

    def _process_bullet_content(self, content):
        """
        Process bullet content with special handling for common patterns.
        This handles several formats:
        - "**Text:** More text" (Bold lead with colon)
        - "Text with **bold words** in the middle"
        """
        # Debug print
        if self.debug_mode:
            print(f"Processing bullet content: {content}")

        # First pattern: "**Text:** More text" 
        pattern1 = re.compile(r'^\s*\*\*([^*:]+)\*\*:\s*(.*)', re.DOTALL)
        match1 = pattern1.match(content)
        if match1:
            bold_part = match1.group(1).strip()
            rest_part = match1.group(2).strip()
            result = f"<b>{bold_part}:</b> {self._process_markdown_inline(rest_part)}"
            if self.debug_mode:
                print(f"Pattern 1 matched. Result: {result}")
            return result
            
        # Second pattern: Handle any inline bold markdown
        result = self._process_markdown_inline(content)
        if self.debug_mode and result != content:
            print(f"Applied inline processing. Result: {result}")
        return result

    def _process_markdown_inline(self, text):
        """Process Markdown-like formatting to HTML for ReportLab."""
        # Original text for comparison
        original = text
        
        # Bold text - **text**
        text = re.sub(r'\*\*([^*]+?)\*\*', r'<b>\1</b>', text)
        
        # Italic text - *text*
        text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<i>\1</i>', text)
        
        # Fix bold text with colon - convert <b>text</b>: to <b>text:</b>
        text = re.sub(r'<b>([^<:]+)</b>:', r'<b>\1:</b>', text)
        
        # Italic with underscore - _text_
        text = re.sub(r'_([^_]+?)_', r'<i>\1</i>', text)
        
        # Debug output
        if self.debug_mode and original != text:
            print(f"Markdown processing: '{original}' -> '{text}'")
            
        return text

    def _clean_text(self, text):
        """Clean text to make it safe for ReportLab."""
        # Fix specific problematic tag patterns before processing
        # Fix the specific nested tags issue with salary bracket text
        text = text.replace("<b>Outlier in 'salariu<i>sb' (Salary Bracket):</b>", 
                        "<b>Outlier in 'salariu sb' (Salary Bracket):</b>")
        
        # More general approach to fix overlapping/improperly nested tags
        text = re.sub(r'(<b>.*?)<i>(.*?)</b>(.*?)</i>', r'\1\2</b><i>\3</i>', text)
        text = re.sub(r'(<i>.*?)<b>(.*?)</i>(.*?)</b>', r'\1\2</i><b>\3</b>', text)
        
        # Process markdown first
        text = self._process_markdown_inline(str(text))
        
        # Extract all HTML tags before escaping
        html_tags = []
        pattern = re.compile(r'<[^>]+>')
        for match in pattern.finditer(text):
            html_tags.append((match.start(), match.end(), match.group()))
        
        # Replace problematic characters with their HTML entities
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        
        # Restore the original HTML tags (in reverse order to preserve positions)
        for start, end, tag in reversed(html_tags):
            replacement_length = end - start
            entity_length = len(text[start:start+replacement_length])
            
            # The length of "&lt;" is different from "<", so adjust positions
            text = text[:start] + tag + text[start+entity_length:]
        
        # Remove any non-printable characters
        text = ''.join(char for char in text if ord(char) > 31 or ord(char) == 9)
        
        # Fix unmatched HTML tags
        open_b = text.count('<b>')
        close_b = text.count('</b>')
        if open_b > close_b:
            text += '</b>' * (open_b - close_b)
        elif close_b > open_b:
            text = '<b>' * (close_b - open_b) + text
            
        open_i = text.count('<i>')
        close_i = text.count('</i>')
        if open_i > close_i:
            text += '</i>' * (open_i - close_i)
        elif close_i > open_i:
            text = '<i>' * (close_i - open_i) + text
            
        return text