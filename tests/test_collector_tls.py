import ssl
import unittest
from unittest.mock import MagicMock, patch

import automatic_collector


class _StreamingResponse:
    def __init__(self, headers, chunks):
        self.headers = headers
        self.chunks = chunks
        self.url = "https://example.test/report.pdf"
        self.status_code = 200
        self.is_redirect = False
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.closed = True
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        return iter(self.chunks)


class _FakePdfPage:
    def __init__(self, text):
        self.text = text

    def get_text(self, text_format):
        return self.text


class _FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        return False

    def __len__(self):
        return len(self.pages)

    def load_page(self, page_index):
        return self.pages[page_index]


class CollectorTlsTests(unittest.TestCase):
    def test_pdf_extraction_includes_appendix_beyond_old_page_and_text_caps(self):
        pages = [
            _FakePdfPage(
                f"Page {page_index:03d} "
                + (
                    "INDEPENDENT ASSURANCE STATEMENT "
                    f"ASSURANCE_APPENDIX_PAGE_{page_index:03d} "
                    if page_index == 104
                    else ""
                )
                + ("x" * 2900)
            )
            for page_index in range(110)
        ]

        with patch.object(
            automatic_collector,
            "download_content",
            return_value=(
                {
                    "final_url": "https://example.test/report.pdf",
                    "status_code": 200,
                    "content_type": "application/pdf",
                },
                b"%PDF",
            ),
        ):
            with patch.object(
                automatic_collector,
                "is_pdf_response",
                return_value=True,
            ):
                with patch.object(
                    automatic_collector.pymupdf,
                    "open",
                    return_value=_FakePdf(pages),
                ):
                    document = automatic_collector.extract_pdf_document(
                        session=object(),
                        url="https://example.test/report.pdf",
                    )

        self.assertEqual(document["status"], "success")
        self.assertEqual(document["page_count"], 110)
        self.assertIn("ASSURANCE_APPENDIX_PAGE_104", document["text"])
        self.assertLessEqual(
            len(document["text"]),
            automatic_collector.MAX_PDF_TEXT_CHARS,
        )

    def test_pdf_size_limit_is_configurable_and_bounded(self):
        with patch.dict(
            "os.environ",
            {automatic_collector.PDF_SIZE_LIMIT_ENV: "3145728"},
        ):
            self.assertEqual(
                automatic_collector._configured_pdf_size_limit(),
                3 * 1024 * 1024,
            )

        with patch.dict(
            "os.environ",
            {
                automatic_collector.PDF_SIZE_LIMIT_ENV: str(
                    automatic_collector.MAX_ALLOWED_PDF_BYTES + 1
                )
            },
        ):
            with self.assertRaisesRegex(ValueError, "must be between"):
                automatic_collector._configured_pdf_size_limit()

    def test_download_content_enforces_limit_without_content_length(self):
        response = _StreamingResponse({}, [b"1234", b"5678"])
        session = MagicMock()
        session.get.return_value = response

        with patch.object(
            automatic_collector,
            "is_allowed_by_robots",
            return_value=True,
        ):
            with self.assertRaisesRegex(ValueError, "7-byte download limit"):
                automatic_collector.download_content(
                    session,
                    "https://example.test/report.pdf",
                    7,
                )

        self.assertTrue(response.closed)

    def test_download_content_enforces_streamed_bytes_when_length_underreports(self):
        response = _StreamingResponse(
            {"Content-Length": "4"},
            [b"1234", b"5678"],
        )
        session = MagicMock()
        session.get.return_value = response

        with patch.object(
            automatic_collector,
            "is_allowed_by_robots",
            return_value=True,
        ):
            with self.assertRaisesRegex(ValueError, "7-byte download limit"):
                automatic_collector.download_content(
                    session,
                    "https://example.test/report.pdf",
                    7,
                )

        self.assertTrue(response.closed)

    def test_failed_pdf_response_has_user_readable_forbidden_reason(self):
        error = automatic_collector.requests.HTTPError(
            "403 Client Error: Forbidden",
        )
        error.http_status = 403
        error.final_url = "https://example.test/report.pdf"

        with patch.object(
            automatic_collector,
            "download_content",
            side_effect=error,
        ):
            document = automatic_collector.extract_pdf_document(
                session=object(),
                url="https://example.test/report.pdf",
            )

        self.assertEqual(document["status"], "failed")
        self.assertEqual(document["http_status"], 403)
        self.assertEqual(
            document["error"],
            "HTTP 403 Forbidden: the source denied this request; "
            "automated collection was not authorized.",
        )

    def test_https_session_uses_system_roots_and_keeps_verification_enabled(self):
        session = automatic_collector.create_session()
        try:
            self.assertIs(session.verify, True)
            adapter = session.get_adapter("https://www.epa.gov/robots.txt")
            context = adapter.poolmanager.connection_pool_kw["ssl_context"]
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            self.assertTrue(context.check_hostname)
            self.assertEqual(
                adapter.max_retries.total,
                automatic_collector.HTTP_RETRY_TOTAL,
            )
            self.assertIn(429, adapter.max_retries.status_forcelist)
            self.assertTrue(adapter.max_retries.respect_retry_after_header)
            http_adapter = session.get_adapter("http://example.test/")
            self.assertEqual(
                http_adapter.max_retries.total,
                automatic_collector.HTTP_RETRY_TOTAL,
            )
        finally:
            session.close()

    def test_robots_check_uses_verified_collector_session(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        response.status_code = 200
        response.text = (
            "User-agent: *\n"
            "Disallow: /blocked/\n"
        )

        session = MagicMock()
        session.__enter__.return_value = session
        session.__exit__.return_value = False
        session.get.return_value = response

        automatic_collector.load_robots_parser.cache_clear()
        try:
            with patch.object(
                automatic_collector,
                "create_session",
                return_value=session,
            ):
                parser = automatic_collector.load_robots_parser(
                    "https://example.test/robots.txt"
                )
        finally:
            automatic_collector.load_robots_parser.cache_clear()

        session.get.assert_called_once_with(
            "https://example.test/robots.txt",
            timeout=automatic_collector.REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        self.assertTrue(
            parser.can_fetch(
                automatic_collector.USER_AGENT,
                "https://example.test/public/",
            )
        )
        self.assertFalse(
            parser.can_fetch(
                automatic_collector.USER_AGENT,
                "https://example.test/blocked/page",
            )
        )


if __name__ == "__main__":
    unittest.main()
