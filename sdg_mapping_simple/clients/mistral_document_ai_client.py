"""Client for Mistral Document AI hosted on Azure AI Foundry."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pymupdf
import requests
from requests import Response, Session

from ._utils import require_setting
from .config import get_settings


DEFAULT_MODEL = "mistral-document-ai-2505"
DEFAULT_TIMEOUT = 120.0  # Document processing can take longer
MAX_PAGES_BASIC = 30
MAX_PAGES_DOCUMENT_ANNOTATION = 8
MAX_DOCUMENT_SIZE_MB = 30


class DocumentType(str, Enum):
    """Supported document types for Mistral Document AI."""

    IMAGE_URL = "image_url"
    DOCUMENT_URL = "document_url"


class AnnotationType(str, Enum):
    """Types of annotations supported."""

    BBOX = "bbox_annotation"
    DOCUMENT = "document_annotation"


class MistralDocumentAIError(RuntimeError):
    """Error raised when a Mistral Document AI API call fails."""

    def __init__(self, message: str, *, status_code: int, detail: Any | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


@dataclass(slots=True)
class MistralDocumentAIClient:
    """Client for Mistral Document AI on Azure AI Foundry.

    This client provides document OCR and structured extraction capabilities.
    It handles base64 encoding of documents and enforces page limits.

    Limitations:
    - Documents: max 30MB and 30 pages for basic OCR
    - Document annotations: max 8 pages
    - Requires AZURE_API_KEY and AZURE_API_ENDPOINT environment variables
    """

    api_key: str
    endpoint: str
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    include_image_base64: bool = True
    session: Session | None = field(default=None, repr=False)
    _session: Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.endpoint = self.endpoint.rstrip("/")
        object.__setattr__(self, "_session", self.session or requests.Session())
        if not hasattr(self._session, "headers") or self._session.headers is None:
            self._session.headers = {}  # type: ignore[assignment]
        self._session.headers.setdefault("Content-Type", "application/json")
        self._session.headers["Authorization"] = f"Bearer {self.api_key}"

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    def __enter__(self) -> MistralDocumentAIClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @staticmethod
    def _encode_file_to_base64(file_path_or_url: str) -> str:
        """Encode a local file or download and encode a URL to base64.

        Args:
            file_path_or_url: Local file path or URL to a document

        Returns:
            Base64 encoded string of the file contents

        Raises:
            MistralDocumentAIError: If file cannot be read or downloaded
        """
        # Check if it's a URL or local path
        parsed = urlparse(file_path_or_url)
        is_url = parsed.scheme in ("http", "https")

        try:
            if is_url:
                # Download the file
                response = requests.get(file_path_or_url, timeout=60)
                response.raise_for_status()
                content = response.content
            else:
                # Read local file
                with open(file_path_or_url, "rb") as f:
                    content = f.read()

            # Check size limit
            size_mb = len(content) / (1024 * 1024)
            if size_mb > MAX_DOCUMENT_SIZE_MB:
                raise MistralDocumentAIError(
                    f"Document size ({size_mb:.2f}MB) exceeds maximum allowed size ({MAX_DOCUMENT_SIZE_MB}MB)",
                    status_code=400,
                )

            return base64.b64encode(content).decode("utf-8")
        except requests.RequestException as exc:
            raise MistralDocumentAIError(
                f"Failed to download document from {file_path_or_url}",
                status_code=0,
                detail=str(exc),
            ) from exc
        except OSError as exc:
            raise MistralDocumentAIError(
                f"Failed to read local file {file_path_or_url}",
                status_code=0,
                detail=str(exc),
            ) from exc

    @staticmethod
    def _detect_mime_type(file_path_or_url: str) -> tuple[DocumentType, str]:
        """Detect MIME type from file extension.

        Args:
            file_path_or_url: File path or URL

        Returns:
            Tuple of (DocumentType, mime_type)
        """
        lower_path = file_path_or_url.lower()

        # Image formats
        if any(lower_path.endswith(ext) for ext in [".jpg", ".jpeg"]):
            return DocumentType.IMAGE_URL, "image/jpeg"
        if lower_path.endswith(".png"):
            return DocumentType.IMAGE_URL, "image/png"
        if lower_path.endswith(".avif"):
            return DocumentType.IMAGE_URL, "image/avif"

        # Document formats
        if lower_path.endswith(".pdf"):
            return DocumentType.DOCUMENT_URL, "application/pdf"
        if lower_path.endswith(".pptx"):
            return DocumentType.DOCUMENT_URL, "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        if lower_path.endswith(".docx"):
            return DocumentType.DOCUMENT_URL, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

        # Default to PDF
        return DocumentType.DOCUMENT_URL, "application/pdf"

    @staticmethod
    def _get_pdf_page_count(file_path_or_url: str) -> int | None:
        """Get the number of pages in a PDF document.

        Args:
            file_path_or_url: Path to local PDF file or URL

        Returns:
            Number of pages in the PDF, or None if not a PDF or cannot be read
        """
        # Only works for local files, not URLs
        parsed = urlparse(file_path_or_url)
        if parsed.scheme in ("http", "https"):
            return None

        # Only works for PDFs
        if not file_path_or_url.lower().endswith(".pdf"):
            return None

        try:
            pdf_path = Path(file_path_or_url)
            if not pdf_path.exists():
                return None

            with pymupdf.open(pdf_path) as doc:
                return len(doc)
        except Exception:
            # If we can't read the PDF, return None and let the API handle it
            return None

    def _make_request(self, payload: dict[str, Any]) -> Any:
        """Make a request to the Mistral Document AI API.

        Args:
            payload: Request payload

        Returns:
            API response as dict

        Raises:
            MistralDocumentAIError: If the request fails
        """
        try:
            response = self._session.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise MistralDocumentAIError(
                "Mistral Document AI API request failed",
                status_code=0,
                detail=str(exc),
            ) from exc

        return self._handle_response(response)

    def _handle_response(self, response: Response) -> Any:
        """Handle API response and raise errors if needed.

        Args:
            response: HTTP response

        Returns:
            Parsed JSON response

        Raises:
            MistralDocumentAIError: If response indicates an error
        """
        if response.ok:
            if response.status_code == 204 or not response.content:
                return None
            if "application/json" in response.headers.get("Content-Type", ""):
                try:
                    return response.json()
                except ValueError as exc:
                    raise MistralDocumentAIError(
                        "Failed to decode JSON from Mistral response",
                        status_code=response.status_code,
                        detail=response.text,
                    ) from exc
            return response.text

        detail: Any | None = None
        message = f"Mistral Document AI API responded with status {response.status_code}"

        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            try:
                detail = response.json()
                if isinstance(detail, dict):
                    error = detail.get("error") or detail.get("message")
                    if error:
                        message = str(error)
            except ValueError:
                detail = response.text
        else:
            detail = response.text

        raise MistralDocumentAIError(message, status_code=response.status_code, detail=detail)

    def extract_text(
        self,
        document_path_or_url: str,
        *,
        include_image_base64: bool | None = None,
    ) -> dict[str, Any]:
        """Extract text from a document using basic OCR.

        Args:
            document_path_or_url: Path to local file or URL to document
            include_image_base64: Whether to include base64 of extracted images
                                 (defaults to client setting)

        Returns:
            OCR result containing extracted text and metadata

        Raises:
            MistralDocumentAIError: If extraction fails

        Example:
            >>> client = get_mistral_document_ai_client()
            >>> result = client.extract_text("path/to/document.pdf")
            >>> print(result["text"])
        """
        # Encode document
        base64_content = self._encode_file_to_base64(document_path_or_url)
        doc_type, mime_type = self._detect_mime_type(document_path_or_url)

        # Build data URL
        data_url = f"data:{mime_type};base64,{base64_content}"

        # Build payload
        payload: dict[str, Any] = {
            "model": self.model,
            "document": {
                "type": doc_type.value,
                doc_type.value: data_url,
            },
            "include_image_base64": (
                include_image_base64 if include_image_base64 is not None else self.include_image_base64
            ),
        }

        return self._make_request(payload)

    def extract_text_to_file(
        self,
        document_path_or_url: str,
        output_file: str = "extracted_text.txt",
        *,
        include_image_base64: bool | None = None,
    ) -> str:
        """Extract text from a document and save to a text file.

        Args:
            document_path_or_url: Path to local file or URL to document
            output_file: Path to output text file (default: extracted_text.txt)
            include_image_base64: Whether to include base64 of extracted images

        Returns:
            Path to the saved text file

        Raises:
            MistralDocumentAIError: If extraction fails
        """
        # Extract text using existing method
        result = self.extract_text(document_path_or_url, include_image_base64=include_image_base64)
        
        # Parse the result to get text
        full_text = ""
        
        if isinstance(result, dict):
            if "text" in result:
                full_text = result["text"]
            elif "pages" in result:
                # Concatenate text from all pages
                texts = []
                for i, page in enumerate(result["pages"]):
                    texts.append(f"\n--- Page {i+1} ---\n")
                    if "markdown" in page:
                        texts.append(page["markdown"])
                    elif "text" in page:
                        texts.append(page["text"])
                full_text = "\n".join(texts)
        else:
            full_text = str(result)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        print(f"✅ Text extracted and saved to: {output_file}")
        print(f"📄 Total characters: {len(full_text)}")
        
        return output_file

    def extract_structured(
        self,
        document_path_or_url: str,
        *,
        schema: dict[str, Any],
        annotation_type: AnnotationType = AnnotationType.DOCUMENT,
        include_image_base64: bool | None = None,
    ) -> dict[str, Any]:
        """Extract structured data from a document using annotations.

        Args:
            document_path_or_url: Path to local file or URL to document
            schema: JSON schema defining the structure to extract
            annotation_type: Type of annotation (BBOX or DOCUMENT)
                           For PDFs with >8 pages, DOCUMENT type will automatically
                           fallback to BBOX to avoid API limitations.
            include_image_base64: Whether to include base64 of extracted images

        Returns:
            Structured extraction result

        Raises:
            MistralDocumentAIError: If extraction fails or page limits exceeded

        Example:
            >>> schema = {
            ...     "properties": {
            ...         "title": {"type": "string"},
            ...         "date": {"type": "string"}
            ...     },
            ...     "required": ["title"],
            ...     "type": "object"
            ... }
            >>> result = client.extract_structured("doc.pdf", schema=schema)
        """
        # Check page limit for document annotations
        effective_annotation_type = annotation_type
        if annotation_type == AnnotationType.DOCUMENT:
            page_count = self._get_pdf_page_count(document_path_or_url)
            if page_count is not None and page_count > MAX_PAGES_DOCUMENT_ANNOTATION:
                # Automatically fall back to BBOX annotation for large documents
                effective_annotation_type = AnnotationType.BBOX
                # Note: This is a silent fallback. The API will work, but results may differ

        # Encode document
        base64_content = self._encode_file_to_base64(document_path_or_url)
        doc_type, mime_type = self._detect_mime_type(document_path_or_url)

        # Build data URL
        data_url = f"data:{mime_type};base64,{base64_content}"

        # Build annotation format
        annotation_format = {
            "type": "json_schema",
            "json_schema": {
                "schema": schema,
                "name": f"{effective_annotation_type.value}_schema",
                "strict": True,
            },
        }

        # Build payload
        payload: dict[str, Any] = {
            "model": self.model,
            "document": {
                "type": doc_type.value,
                doc_type.value: data_url,
            },
            "include_image_base64": (
                include_image_base64 if include_image_base64 is not None else self.include_image_base64
            ),
        }

        # Add appropriate annotation format
        if effective_annotation_type == AnnotationType.BBOX:
            payload["bbox_annotation_format"] = annotation_format
        else:
            payload["document_annotation_format"] = annotation_format

        return self._make_request(payload)

    def extract_with_prompt(
        self,
        document_path_or_url: str,
        *,
        properties: dict[str, dict[str, str]],
        required: list[str] | None = None,
        annotation_type: AnnotationType = AnnotationType.DOCUMENT,
        include_image_base64: bool | None = None,
    ) -> dict[str, Any]:
        """Extract data from document using a simplified schema definition.

        This is a convenience method that builds the JSON schema from a simpler format.

        Args:
            document_path_or_url: Path to local file or URL to document
            properties: Dict mapping field names to property definitions.
                       Each property should have "type" and optionally "description".
            required: List of required field names
            annotation_type: Type of annotation (BBOX or DOCUMENT)
            include_image_base64: Whether to include base64 of extracted images

        Returns:
            Structured extraction result

        Example:
            >>> properties = {
            ...     "document_type": {
            ...         "title": "Document Type",
            ...         "description": "The type of the document",
            ...         "type": "string"
            ...     },
            ...     "summary": {
            ...         "title": "Summary",
            ...         "description": "Summarize the document",
            ...         "type": "string"
            ...     }
            ... }
            >>> result = client.extract_with_prompt(
            ...     "doc.pdf",
            ...     properties=properties,
            ...     required=["document_type", "summary"]
            ... )
        """
        schema = {
            "properties": properties,
            "required": required or list(properties.keys()),
            "type": "object",
            "additionalProperties": False,
        }

        return self.extract_structured(
            document_path_or_url,
            schema=schema,
            annotation_type=annotation_type,
            include_image_base64=include_image_base64,
        )


def get_mistral_document_ai_client(
    *,
    session: Session | None = None,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT,
    include_image_base64: bool = True,
) -> MistralDocumentAIClient:
    """Create a MistralDocumentAIClient using environment settings.

    Requires AZURE_API_KEY and either MISTRAL_DOCUMENT_AI_ENDPOINT or
    a base endpoint that will have the Mistral path appended.

    Environment variables:
    - AZURE_API_KEY: Your Azure API key (required)
    - MISTRAL_DOCUMENT_AI_ENDPOINT: Full Mistral Document AI endpoint URL
      (e.g., https://your-resource.services.ai.azure.com/providers/mistral/azure/ocr)

    Args:
        session: Optional requests Session to reuse
        model: Model name (default: mistral-document-ai-2505)
        timeout: Request timeout in seconds
        include_image_base64: Whether to include base64 of images by default

    Returns:
        Configured MistralDocumentAIClient

    Raises:
        RuntimeError: If required environment variables are missing

    Example:
        >>> client = get_mistral_document_ai_client()
        >>> result = client.extract_text("path/to/document.pdf")
    """
    settings = get_settings()
    api_key = require_setting(settings.azure_api_key, name="AZURE_API_KEY")
    
    # Use dedicated Mistral endpoint if provided
    endpoint = settings.mistral_document_ai_endpoint
    
    if not endpoint:
        raise RuntimeError(
            "MISTRAL_DOCUMENT_AI_ENDPOINT environment variable is required. "
            "Set it to your full Mistral Document AI endpoint URL, e.g.: "
            "https://your-resource.services.ai.azure.com/providers/mistral/azure/ocr"
        )

    return MistralDocumentAIClient(
        api_key=api_key,
        endpoint=endpoint,
        model=model,
        timeout=timeout,
        include_image_base64=include_image_base64,
        session=session,
    )

