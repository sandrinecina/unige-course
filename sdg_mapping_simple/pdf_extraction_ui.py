"""Streamlit UI for testing Mistral Document AI PDF extraction.

This UI allows you to:
1. Input a PDF URL
2. Preview the PDF
3. Extract content using different methods
4. View results in a human-readable format
"""

from __future__ import annotations

import json
import re
from typing import Any

import streamlit as st
from streamlit.components.v1 import iframe

from clients import get_mistral_document_ai_client
from clients.mistral_document_ai_client import (
    AnnotationType,
    MistralDocumentAIError,
)


def ensure_session_state() -> None:
    """Initialize session state variables."""
    st.session_state.setdefault("pdf_url", "")
    st.session_state.setdefault("extraction_result", None)
    st.session_state.setdefault("extraction_error", None)
    st.session_state.setdefault("extraction_type", "basic")


def parse_api_response(result: dict[str, Any]) -> dict[str, Any] | str:
    """Parse the API response to extract the actual content.
    
    The API response structure can vary:
    - Direct fields like "text", "annotation", "pages"
    - Nested in choices[0].message.content
    
    Args:
        result: Raw API response
        
    Returns:
        Parsed content as dict or string
    """
    if not isinstance(result, dict):
        return result
    
    # Check for pages structure first (contains images, text, etc.)
    if "pages" in result:
        return result
    
    # For basic OCR - look for text field
    if "text" in result:
        return {"text": result["text"], "pages": result.get("pages")}
    
    # For structured extraction - look for annotation
    if "annotation" in result:
        return result["annotation"]
    
    # Check if wrapped in choices structure
    if "choices" in result and len(result["choices"]) > 0:
        choice = result["choices"][0]
        if "message" in choice:
            content = choice["message"].get("content")
            if content:
                # Try to parse as JSON if it's a string
                if isinstance(content, str):
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        return content
                return content
    
    # Return the full result if we can't parse it
    return result


def run_basic_extraction(pdf_url: str) -> None:
    """Run basic OCR text extraction."""
    st.session_state.extraction_error = None
    st.session_state.extraction_result = None
    
    with st.spinner("Extracting text from PDF..."):
        try:
            with get_mistral_document_ai_client() as client:
                result = client.extract_text(pdf_url)
            
            parsed_result = parse_api_response(result)
            st.session_state.extraction_result = {
                "type": "basic",
                "data": parsed_result,
                "raw": result,
            }
        except MistralDocumentAIError as e:
            st.session_state.extraction_error = f"API Error ({e.status_code}): {e}"
        except Exception as e:
            st.session_state.extraction_error = f"Error: {e}"


def run_structured_extraction(pdf_url: str, properties: dict[str, dict[str, Any]]) -> None:
    """Run structured data extraction with custom properties."""
    st.session_state.extraction_error = None
    st.session_state.extraction_result = None
    
    with st.spinner("Extracting structured data from PDF..."):
        try:
            with get_mistral_document_ai_client() as client:
                result = client.extract_with_prompt(
                    pdf_url,
                    properties=properties,
                    required=list(properties.keys()),
                    annotation_type=AnnotationType.DOCUMENT,
                )
            
            parsed_result = parse_api_response(result)
            st.session_state.extraction_result = {
                "type": "structured",
                "data": parsed_result,
                "raw": result,
            }
        except MistralDocumentAIError as e:
            st.session_state.extraction_error = f"API Error ({e.status_code}): {e}"
        except Exception as e:
            st.session_state.extraction_error = f"Error: {e}"


def render_pdf_preview(pdf_url: str) -> None:
    """Render PDF preview."""
    if not pdf_url:
        st.info("👆 Enter a PDF URL in the sidebar to get started")
        return
    
    st.subheader("📄 PDF Preview")
    st.markdown(f"**URL:** [{pdf_url}]({pdf_url})")
    
    with st.expander("Show PDF Preview", expanded=True):
        st.caption("⚠️ Some PDF hosts block embedding. If preview doesn't work, use the link above.")
        try:
            # Try to embed the PDF
            iframe(pdf_url, height=600, scrolling=True)
        except Exception as e:
            st.warning(f"PDF preview unavailable: {str(e)}")


def process_markdown_with_images(markdown: str, images: list[dict[str, Any]]) -> str:
    """Replace image references in markdown with actual base64 data URLs.
    
    Args:
        markdown: Markdown text with image references
        images: List of image objects with id and image_base64
        
    Returns:
        Markdown with embedded images
    """
    if not markdown or not images:
        return markdown
    
    # Create a mapping of image IDs to their base64 data
    image_map = {}
    for img in images:
        img_id = img.get("id", "")
        image_base64 = img.get("image_base64", "")
        if img_id and image_base64:
            image_map[img_id] = image_base64
    
    # Replace image references in markdown
    # Markdown format: ![alt](img-id) or ![](img-id)
    # Pattern to match markdown images: ![...](image-id)
    pattern = r'!\[(.*?)\]\((.*?)\)'
    
    def replace_image(match):
        alt_text = match.group(1)
        img_ref = match.group(2)
        
        # Look up the image in our map
        if img_ref in image_map:
            # Return markdown with the full data URL
            return f'![{alt_text or img_ref}]({image_map[img_ref]})'
        
        # If not found, keep original
        return match.group(0)
    
    return re.sub(pattern, replace_image, markdown)


def render_images_metadata(images: list[dict[str, Any]], page_index: int = 0) -> None:
    """Render metadata about images (collapsed by default)."""
    if not images:
        return
    
    with st.expander(f"🔍 Image Metadata ({len(images)} images)", expanded=False):
        for idx, img in enumerate(images):
            img_id = img.get("id", "unknown")
            
            st.markdown(f"**{idx + 1}. {img_id}**")
            
            # Show coordinates
            top_left_x = img.get("top_left_x", 0)
            top_left_y = img.get("top_left_y", 0)
            bottom_right_x = img.get("bottom_right_x", 0)
            bottom_right_y = img.get("bottom_right_y", 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"Position: ({top_left_x}, {top_left_y}) → ({bottom_right_x}, {bottom_right_y})")
            with col2:
                width = bottom_right_x - top_left_x
                height = bottom_right_y - top_left_y
                st.caption(f"Size: {width} × {height} px")
            
            annotation = img.get("image_annotation")
            if annotation:
                st.markdown("**Annotation:**")
                if isinstance(annotation, dict):
                    st.json(annotation)
                else:
                    st.write(annotation)
            
            if idx < len(images) - 1:
                st.markdown("---")


def render_document_view(data: dict[str, Any]) -> None:
    """Render document-like view with pages, text, and images."""
    pages = data.get("pages", [])
    
    if not pages:
        st.warning("No pages found in response")
        return
    
    st.markdown(f"### 📑 Extracted Document ({len(pages)} pages)")
    st.caption("Scroll down to see all content with embedded images")
    
    # Render all pages in a scrollable view
    for i, page in enumerate(pages):
        page_index = page.get("index", i)
        images = page.get("images", [])
        markdown = page.get("markdown", "")
        text = page.get("text", "")
        
        # Page separator
        st.markdown("---")
        st.markdown(f"#### 📄 Page {page_index + 1}")
        
        # Priority 1: Show markdown with embedded images (the main content!)
        if markdown:
            # Process markdown to embed actual images
            processed_markdown = process_markdown_with_images(markdown, images)
            st.markdown(processed_markdown, unsafe_allow_html=True)
            
            # Show image metadata in collapsed section
            if images:
                render_images_metadata(images, page_index)
        
        # Priority 2: If no markdown, show text
        elif text:
            st.markdown("**📝 Text Content:**")
            st.text_area(
                f"Page {page_index + 1} text",
                value=text,
                height=300,
                label_visibility="collapsed",
                key=f"text_page_{page_index}",
            )
            
            # Show image metadata if available
            if images:
                render_images_metadata(images, page_index)
        
        # Priority 3: If no text or markdown, show raw images
        else:
            if images:
                st.warning("No text/markdown found, showing images only")
                render_images_metadata(images, page_index)


def render_extraction_result() -> None:
    """Render extraction results in a human-readable format."""
    if st.session_state.extraction_error:
        st.error(st.session_state.extraction_error)
        return
    
    if st.session_state.extraction_result is None:
        st.info("👈 Configure extraction settings and click **Extract** to see results")
        return
    
    result = st.session_state.extraction_result
    result_type = result["type"]
    data = result["data"]
    raw = result["raw"]
    
    st.subheader("✨ Extraction Results")
    
    # Add copy button for raw response
    col1, col2 = st.columns([3, 1])
    with col2:
        raw_json = json.dumps(raw, indent=2, ensure_ascii=False)
        st.download_button(
            label="📋 Copy Raw JSON",
            data=raw_json,
            file_name="extraction_result.json",
            mime="application/json",
        )
    
    # Show results based on type
    if result_type == "basic":
        # Check if we have page structure with images
        if isinstance(data, dict) and "pages" in data:
            render_document_view(data)
        elif isinstance(data, dict) and "text" in data:
            # Simple text extraction
            text = data["text"]
            pages = data.get("pages", "Unknown")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Pages Processed", pages if isinstance(pages, int) else "N/A")
            with col2:
                st.metric("Characters Extracted", len(text))
            
            st.markdown("**Extracted Text:**")
            st.text_area(
                "Full Text",
                value=text,
                height=400,
                label_visibility="collapsed",
            )
        else:
            # Fallback to document view if structure is different
            if isinstance(data, dict):
                render_document_view(data)
            else:
                st.json(data)
    
    elif result_type == "structured":
        # Structured extraction results
        st.markdown("**Structured Data:**")
        
        if isinstance(data, dict):
            # Display each field nicely
            for key, value in data.items():
                st.markdown(f"**{key.replace('_', ' ').title()}:**")
                if isinstance(value, (list, dict)):
                    st.json(value)
                else:
                    st.write(value)
                st.markdown("---")
        else:
            st.json(data)
    
    # Show raw response in expander
    with st.expander("🔍 View Raw API Response"):
        st.code(raw_json, language="json")
        st.caption("💡 Use the 'Copy Raw JSON' button above to copy to clipboard")


def main() -> None:
    """Main Streamlit app."""
    st.set_page_config(
        page_title="PDF Extraction Test",
        page_icon="📄",
        layout="wide",
    )
    
    ensure_session_state()
    
    st.title("📄 PDF Extraction Test Tool")
    st.markdown(
        "Test the Mistral Document AI service by extracting content from PDF URLs. "
        "Choose between basic OCR or structured extraction."
    )
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # PDF URL input
        pdf_url = st.text_input(
            "PDF URL",
            value=st.session_state.pdf_url,
            placeholder="https://example.com/document.pdf",
            help="Enter a publicly accessible PDF URL",
            key="pdf_url_input",
        )
        
        # Quick sample button
        if st.button("📎 Load Sample PDF", use_container_width=True):
            st.session_state.pdf_url = "https://www.africau.edu/images/default/sample.pdf"
            st.rerun()
        
        # Update session state when URL changes
        if pdf_url != st.session_state.pdf_url:
            st.session_state.pdf_url = pdf_url
            # Trigger rerun to show preview
            st.rerun()
        
        st.divider()
        
        # Extraction type
        st.header("🔧 Extraction Type")
        extraction_type = st.radio(
            "Select extraction method",
            options=["basic", "structured"],
            format_func=lambda x: "Basic OCR" if x == "basic" else "Structured Extraction",
            help="Basic OCR extracts all text. Structured extraction uses AI to extract specific fields.",
        )
        
        st.session_state.extraction_type = extraction_type
        
        # Configuration for structured extraction
        properties = None
        if extraction_type == "structured":
            st.markdown("**Define fields to extract:**")
            
            # Provide some preset options
            preset = st.selectbox(
                "Use preset",
                options=["Custom", "Annual Report", "Invoice", "General Document"],
            )
            
            if preset == "Annual Report":
                properties = {
                    "organization_name": {
                        "title": "Organization Name",
                        "description": "Name of the organization",
                        "type": "string",
                    },
                    "year": {
                        "title": "Year",
                        "description": "Report year",
                        "type": "string",
                    },
                    "summary": {
                        "title": "Summary",
                        "description": "Brief summary of the report (2-3 sentences)",
                        "type": "string",
                    },
                    "key_achievements": {
                        "title": "Key Achievements",
                        "description": "Main achievements mentioned in the report",
                        "type": "string",
                    },
                    "financial_highlights": {
                        "title": "Financial Highlights",
                        "description": "Key financial information or highlights",
                        "type": "string",
                    },
                }
            elif preset == "Invoice":
                properties = {
                    "invoice_number": {
                        "title": "Invoice Number",
                        "description": "Invoice or reference number",
                        "type": "string",
                    },
                    "date": {
                        "title": "Date",
                        "description": "Invoice date",
                        "type": "string",
                    },
                    "amount": {
                        "title": "Amount",
                        "description": "Total amount",
                        "type": "string",
                    },
                    "vendor": {
                        "title": "Vendor",
                        "description": "Vendor or supplier name",
                        "type": "string",
                    },
                }
            elif preset == "General Document":
                properties = {
                    "document_type": {
                        "title": "Document Type",
                        "description": "Type of document (e.g., report, letter, form)",
                        "type": "string",
                    },
                    "title": {
                        "title": "Title",
                        "description": "Main title or heading",
                        "type": "string",
                    },
                    "summary": {
                        "title": "Summary",
                        "description": "Brief summary of the document content",
                        "type": "string",
                    },
                    "key_topics": {
                        "title": "Key Topics",
                        "description": "Main topics discussed (comma-separated)",
                        "type": "string",
                    },
                }
            else:  # Custom
                st.info("💡 Using Annual Report preset as default. Modify below or choose another preset.")
                properties = {
                    "summary": {
                        "title": "Summary",
                        "description": "A brief summary of the document",
                        "type": "string",
                    },
                    "key_information": {
                        "title": "Key Information",
                        "description": "Important information from the document",
                        "type": "string",
                    },
                }
            
            # Show current schema
            with st.expander("📋 View extraction schema"):
                st.json(properties)
        
        st.divider()
        
        # Extract button
        if st.button("🚀 Extract", type="primary", use_container_width=True):
            if not pdf_url:
                st.error("Please enter a PDF URL first!")
            else:
                if extraction_type == "basic":
                    run_basic_extraction(pdf_url)
                else:
                    if properties:
                        run_structured_extraction(pdf_url, properties)
                    else:
                        st.error("Please define properties for structured extraction")
    
    # Main content area
    if st.session_state.pdf_url:
        render_pdf_preview(st.session_state.pdf_url)
    
    render_extraction_result()
    
    # Help section
    with st.expander("ℹ️ Help & Information"):
        st.markdown("""
        ### How to use this tool
        
        1. **Enter a PDF URL** in the sidebar (or click "Load Sample PDF")
        2. **Preview appears immediately** - check if your PDF is accessible
        3. **Choose extraction type:**
           - **Basic OCR**: Extracts structured markdown with embedded images
           - **Structured Extraction**: Uses AI to extract specific fields you define
        4. **Click Extract** to process the document
        5. **Scroll through results** - images appear inline where they belong
        6. **Copy raw JSON** using the download button
        
        ### Limitations
        
        - Maximum file size: 30 MB
        - Maximum pages for basic OCR: 30 pages
        - Maximum pages for structured extraction: 8 pages (automatic fallback for larger docs)
        - PDF preview may not work if the host blocks embedding
        
        ### Features
        
        - ✅ **Markdown Display** - document content shown as readable markdown
        - ✅ **Inline Images** - images appear exactly where they belong in the document
        - ✅ **Live PDF Preview** - see the original as you add the URL
        - ✅ **Scrollable View** - browse all pages naturally without clicking
        - ✅ **Image Metadata** - coordinates, sizes, and annotations (collapsed)
        - ✅ **Copy to Clipboard** - download raw JSON response
        - ✅ **Preset Templates** - quick configs for common document types
        
        ### Tips
        
        - **Markdown is primary** - shows the document structure with embedded images
        - **Just scroll** - all pages appear one after another, no tabs or dropdowns
        - Images appear inline in the markdown at their original positions
        - Expand "Image Metadata" to see technical details (coordinates, sizes, annotations)
        - Use the raw API response to see the complete data structure
        - For large documents, consider using basic OCR instead of structured extraction
        """)


if __name__ == "__main__":
    main()

