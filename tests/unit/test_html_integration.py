from unittest.mock import MagicMock, patch
from pathlib import Path
from app.graph.nodes import query_understanding, fetch_pdf, parse_pdf, retry_parse
from app.pdf.extractor import ExtractedBlock
from app.pdf.html_extractor import extract_html

def test_query_understanding_html_routes_as_paper():
    state = {"user_input": "https://arxiv.org/html/2603.13942v3"}
    res = query_understanding(state)
    assert res["query_type"] == "paper"
    assert res["arxiv_id"] == "2603.13942"
    assert res["input_kind"] == "html_url"
    assert res["paper_version"] == "v3"

def test_html_extractor_common_extracted_block():
    # Simple test to verify the extractor returns ExtractedBlock objects
    # We will mock the requests.get
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <div id="abstract">This is the abstract.</div>
            <article>
                <h1>Introduction</h1>
                <p>This is a long paragraph that is definitely over 30 characters in length to be extracted.</p>
                <h2>Background</h2>
                <p>This is another paragraph that is over thirty characters long for extraction.</p>
            </article>
        </html>
        """
        mock_get.return_value = mock_response
        blocks = extract_html("http://fake")
        assert len(blocks) == 3
        assert isinstance(blocks[0], ExtractedBlock)
        assert blocks[0].text == "This is the abstract."
        assert blocks[0].section == "Abstract"
        assert blocks[0].page == 1
        
        assert blocks[1].section == "Introduction"
        assert blocks[1].text == "This is a long paragraph that is definitely over 30 characters in length to be extracted."
        
        assert blocks[2].section == "Background"

def test_fetch_pdf_caches_and_does_not_redownload(tmp_path):
    services = MagicMock()
    services.settings.data_dir = tmp_path
    (tmp_path / "papers").mkdir()
    
    paper = MagicMock()
    paper.paper_id = "1234"
    paper.pdf_url = "http://fake"
    paper.html_url = "http://html"
    
    path = tmp_path / "papers" / "1234.pdf"
    path.write_bytes(b"A" * 15000) # Valid size cache
    
    state = {"selected_paper": paper}
    
    with patch("app.graph.nodes.download_pdf") as mock_download:
        res = fetch_pdf(services)(state)
        mock_download.assert_not_called()
        assert res["pdf_path"] == str(path)
        assert res["html_url"] == "http://html"
        
def test_fetch_pdf_downloads_if_too_small(tmp_path):
    services = MagicMock()
    services.settings.data_dir = tmp_path
    (tmp_path / "papers").mkdir()
    
    paper = MagicMock()
    paper.paper_id = "1234"
    paper.pdf_url = "http://fake"
    
    path = tmp_path / "papers" / "1234.pdf"
    path.write_bytes(b"A" * 500) # Too small
    
    state = {"selected_paper": paper}
    
    with patch("app.graph.nodes.download_pdf") as mock_download:
        res = fetch_pdf(services)(state)
        mock_download.assert_called_once()
        
@patch("app.pdf.html_extractor.extract_html")
@patch("app.graph.nodes.extract_pdf")
def test_parse_pdf_uses_html_then_fallback(mock_pdf, mock_html):
    mock_html.return_value = ["html_block"]
    mock_pdf.return_value = ["pdf_block"]
    
    # Has HTML URL -> Uses HTML
    state = {"pdf_path": "fake", "html_url": "http://fake"}
    res = parse_pdf(state)
    assert res["parsed_sections"] == ["html_block"]
    assert res["source_type"] == "html"
    assert res["parse_valid"] is True
    mock_pdf.assert_not_called()
    
    # HTML fails -> Fallback to PDF
    mock_html.side_effect = Exception("fail")
    res = parse_pdf(state)
    assert res["parsed_sections"] == ["pdf_block"]
    assert res["source_type"] == "pdf"
    
    # Retry always goes to PDF
    mock_pdf.reset_mock()
    res = retry_parse(state)
    assert res["parsed_sections"] == ["pdf_block"]
    assert res["source_type"] == "pdf"
    assert res["retry_count"] == 1
    
