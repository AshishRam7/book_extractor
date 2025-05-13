import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const API_BASE_URL = 'http://localhost:8001';

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]); // Changed to array for multiple files
  const [pageRanges, setPageRanges] = useState({}); // Stores { fileIndex: "range_string" }
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState('');
  const [jobMessage, setJobMessage] = useState('');
  const [jobResult, setJobResult] = useState(null);
  const [selectedChapterCheckboxes, setSelectedChapterCheckboxes] = useState({});
  const [retrievedMarkdown, setRetrievedMarkdown] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const pollingIntervalRef = useRef(null);
  const fileInputRef = useRef(null); // Ref for file input

  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) clearInterval(pollingIntervalRef.current);
    };
  }, []);

  const resetUIForNewJob = (clearFiles = false) => {
    setJobId(null);
    setJobStatus('');
    setJobMessage('');
    setJobResult(null);
    setSelectedChapterCheckboxes({});
    setRetrievedMarkdown('');
    setError('');
    setIsLoading(false);
    if (pollingIntervalRef.current) clearInterval(pollingIntervalRef.current);
    if (clearFiles) {
        setSelectedFiles([]);
        setPageRanges({});
        if(fileInputRef.current) fileInputRef.current.value = ""; // Clear file input display
    }
  };

  const handleFileChange = (event) => {
    const files = Array.from(event.target.files);
    setSelectedFiles(files);
    // Reset page ranges for new set of files
    const newPageRanges = {};
    files.forEach((_, index) => {
      newPageRanges[index] = ""; // Initialize with empty string
    });
    setPageRanges(newPageRanges);
    resetUIForNewJob(false); // Don't clear files here, just job state
  };

  const handlePageRangeChange = (fileIndex, value) => {
    setPageRanges(prev => ({ ...prev, [fileIndex]: value }));
  };

  const handleUploadAndExtract = async () => {
    if (!selectedFiles || selectedFiles.length === 0) {
      setError('Please select PDF files to upload.');
      return;
    }

    resetUIForNewJob(false);
    setIsLoading(true);
    setError('');
    setJobStatus('Uploading and starting job...');

    const formData = new FormData();
    selectedFiles.forEach((file) => {
      formData.append('files', file);
    });

    // Append page_ranges for each file in the correct order
    // Backend expects a list of strings for page_ranges, one for each file
    selectedFiles.forEach((_, index) => {
        formData.append('page_ranges', pageRanges[index] || ""); // Send empty string if no range specified
    });


    try {
      const response = await fetch(`${API_BASE_URL}/extract-chapters`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        setError(data.detail || `Failed to start job (HTTP ${response.status})`);
        setJobStatus('Error');
        setIsLoading(false);
        return;
      }

      setJobId(data.job_id);
      setJobStatus(data.status);
      setJobMessage(data.message);
      setIsLoading(false);
      startPolling(data.job_id);
    } catch (err) {
      console.error('Upload error:', err);
      setError('Network error or server is unreachable during upload.');
      setJobStatus('Error');
      setIsLoading(false);
    }
  };

  const startPolling = (currentJobId) => {
    if (pollingIntervalRef.current) clearInterval(pollingIntervalRef.current);
    fetchJobStatus(currentJobId); 
    pollingIntervalRef.current = setInterval(() => fetchJobStatus(currentJobId), 3000);
  };

  const fetchJobStatus = async (currentJobId) => {
    if (!currentJobId) return;
    try {
      const response = await fetch(`${API_BASE_URL}/status/${currentJobId}`);
      const data = await response.json();
      if (!response.ok) {
        setError(data.detail || `Error fetching status (HTTP ${response.status})`);
        if (response.status === 404) {
            setJobStatus('Error'); setJobMessage('Job not found.');
            if (pollingIntervalRef.current) clearInterval(pollingIntervalRef.current);
        } return;
      }
      setJobStatus(data.status); setJobMessage(data.message); setJobResult(data.result);
      if (data.status === 'completed' || data.status === 'error') {
        if (pollingIntervalRef.current) clearInterval(pollingIntervalRef.current);
        if (data.status === 'completed' && data.result && data.result.all_available_chapters) {
          const initialCheckboxes = {};
          data.result.all_available_chapters.forEach((_, index) => { initialCheckboxes[index] = false; });
          setSelectedChapterCheckboxes(initialCheckboxes);
        }
      }
    } catch (err) { console.error('Polling error:', err); setError('Network error polling status.'); }
  };

  const handleChapterSelectionChange = (chapterIndex) => {
    setSelectedChapterCheckboxes(prev => ({ ...prev, [chapterIndex]: !prev[chapterIndex] }));
  };

  const handleGetSelectedChapters = async () => {
    if (!jobId || !jobResult || !jobResult.all_available_chapters) { setError('Job not complete or no chapters.'); return; }
    const chaptersToRetrieve = jobResult.all_available_chapters
      .filter((_, index) => selectedChapterCheckboxes[index])
      .map(chap => ({ doc_filename: chap.doc_filename, chapter_title: chap.chapter_title }));
    if (chaptersToRetrieve.length === 0) { setError('Please select chapters.'); return; }

    setIsLoading(true); setError(''); setRetrievedMarkdown('Fetching...');
    try {
      const response = await fetch(`${API_BASE_URL}/get-selected-chapters/${jobId}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chapters_to_retrieve: chaptersToRetrieve }),
      });
      const data = await response.json();
      if (!response.ok) { setError(data.detail || 'Failed to retrieve chapters.'); setRetrievedMarkdown(''); }
      else { setRetrievedMarkdown(data.selected_markdown_content || 'No content.'); setJobMessage(prev => `${prev} | ${data.message}`); }
    } catch (err) { console.error('Retrieve error:', err); setError('Network error retrieving chapters.'); setRetrievedMarkdown('');}
    finally { setIsLoading(false); }
  };

  const handleDeleteJob = async () => {
    if (!jobId) { setError("No job ID."); return; }
    if (!window.confirm(`Delete job ${jobId}?`)) return;
    setIsLoading(true); setError('');
    try {
      const response = await fetch(`${API_BASE_URL}/job/${jobId}`, { method: 'DELETE' });
      const data = await response.json(); // Expects JSON response even for DELETE
      if (!response.ok) { setError(data.detail || 'Failed to delete job.'); }
      else { alert(data.message || "Job deleted."); resetUIForNewJob(true); }
    } catch (err) { console.error('Delete error:', err); setError('Network error deleting job.');}
    finally { setIsLoading(false); }
  };

  return (
    <div className="App">
      <header className="App-header"><h1>PDF Chapter Extractor</h1></header>
      <main>
        <section className="upload-section">
          <h2>1. Upload PDFs & Specify Page Ranges (Optional)</h2>
          <input type="file" multiple accept=".pdf" onChange={handleFileChange} ref={fileInputRef}/>
          {selectedFiles.length > 0 && (
            <div className="file-list">
              <h4>Selected Files:</h4>
              {selectedFiles.map((file, index) => (
                <div key={index} className="file-item">
                  <span>{file.name}</span>
                  <input
                    type="text"
                    placeholder="e.g., 1-5, 7, 10-"
                    value={pageRanges[index] || ""}
                    onChange={(e) => handlePageRangeChange(index, e.target.value)}
                    className="page-range-input"
                  />
                </div>
              ))}
            </div>
          )}
          <button onClick={handleUploadAndExtract} disabled={isLoading || selectedFiles.length === 0}>
            {isLoading && jobStatus.startsWith('Uploading') ? 'Processing...' : 'Upload & Extract'}
          </button>
        </section>

        {error && <p className="error-message">Error: {error}</p>}

        {jobId && (
          <section className="job-status-section">
            <h2>2. Job Progress</h2>
            <p><strong>Job ID:</strong> {jobId}</p>
            <p><strong>Status:</strong> {jobStatus}</p>
            <p><strong>Message:</strong> {jobMessage}</p>
            {isLoading && (jobStatus !== 'Uploading and starting job...') && <p>Loading...</p>}
            {jobId && !isLoading && <button onClick={handleDeleteJob} className="delete-button">Delete Job</button>}
          </section>
        )}

        {jobResult && jobResult.all_available_chapters && jobStatus === 'completed' && (
          <section className="chapter-selection-section">
            <h2>3. Select Chapters</h2>
            {jobResult.all_available_chapters.length > 0 ? (
              <ul>{jobResult.all_available_chapters.map((chap, idx) => (
                <li key={`${chap.doc_filename}-${chap.chapter_title}-${idx}`}>
                  <input type="checkbox" id={`chap-${idx}`} checked={selectedChapterCheckboxes[idx]||false} onChange={()=>handleChapterSelectionChange(idx)}/>
                  <label htmlFor={`chap-${idx}`}>{chap.chapter_title} (<i>{chap.doc_filename}</i>)</label>
                </li>))}
              </ul>) : <p>No chapters identified.</p>}
            {jobResult.all_available_chapters.length > 0 &&
              <button onClick={handleGetSelectedChapters} disabled={isLoading}>{isLoading ? 'Fetching...' : 'Get Content'}</button>}
          </section>
        )}

        {retrievedMarkdown && (
          <section className="markdown-display-section">
            <h2>4. Selected Content</h2><pre>{retrievedMarkdown}</pre>
          </section>
        )}

        {jobResult && jobResult.documents && jobStatus === 'completed' && (
          <section className="resource-links-section">
            <h2>5. Resource Links</h2>
            {jobResult.documents.map((doc, idx) => (
              <div key={doc.original_filename + idx}>
                <h4>{doc.original_filename}</h4>
                {doc.processed_markdown_file_url && <p><a href={`${API_BASE_URL}${doc.processed_markdown_file_url}`} target="_blank" rel="noopener noreferrer">Full Markdown</a></p>}
                {doc.image_urls && doc.image_urls.length > 0 && (<>
                  <p>Images:</p><ul>{doc.image_urls.map((url, i) => (
                    <li key={url+i}><a href={`${API_BASE_URL}${url}`} target="_blank" rel="noopener noreferrer">{url.substring(url.lastIndexOf('/') + 1)}</a></li>
                  ))}</ul></>)}
                {doc.chapters && doc.chapters.Error && <p className="error-message">Doc Error: {doc.chapters.Error}</p>}
              </div>))}
          </section>
        )}
      </main>
    </div>
  );
}
export default App;