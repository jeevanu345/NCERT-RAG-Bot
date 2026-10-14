import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import uuid

from fastapi.testclient import TestClient
from fastapi import HTTPException
from pypdf import PdfWriter
from io import BytesIO
import requests

from api.chunking import chunk_text
from api.index import app
from api import rag_pipeline as rag
from pdf_fixture import make_pdf

KEY = 'nvapi-test-placeholder'
MODEL = rag.DEFAULT_MODEL

class TutorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'NCERT_DATA_DIR': self.temp.name, 'NVIDIA_API_KEY': ''})
        self.env.start()
        self.client = TestClient(app)
        self.workspace = str(uuid.uuid4())
        self.headers = {'X-Workspace-ID': self.workspace, 'Authorization': f'Bearer {KEY}'}

    def tearDown(self):
        self.client.close(); self.env.stop(); self.temp.cleanup()

    def upload(self, body=None, name='science.pdf', headers=None):
        return self.client.post('/api/upload', params={'filename': name}, content=body if body is not None else make_pdf(), headers=headers or self.headers)

    def chat(self, question='What is electric current?', **kwargs):
        return self.client.post('/api/chat', json={'messages': [{'role': 'user', 'content': question}], 'model': MODEL, **kwargs}, headers=self.headers)

    def test_chunking_empty_and_bounds(self):
        self.assertEqual(chunk_text('  \n '), [])
        self.assertEqual(chunk_text('short text'), ['short text'])
        for size, overlap in [(0, 0), (10, 10), (10, -1)]:
            with self.assertRaises(ValueError): chunk_text('text', size, overlap)

    def test_chunking_overlap_and_long_words(self):
        original = ' '.join(f'word{i}' for i in range(300))
        chunks = chunk_text(original, 120, 20)
        self.assertTrue(all(0 < len(c) <= 120 for c in chunks))
        self.assertTrue(all(word in ' '.join(chunks) for word in original.split()))
        self.assertTrue(all(chunks[i][-15:] in chunks[i+1][:25] for i in range(len(chunks)-1)))
        self.assertEqual(len(chunk_text('x' * 1000, 100, 10)), 11)

    def test_health_does_not_expose_key(self):
        with patch.dict(os.environ, {'NVIDIA_API_KEY': KEY}):
            response = self.client.get('/api/health')
        self.assertTrue(response.json()['server_key_configured'])
        self.assertNotIn(KEY, response.text)

    def test_upload_extracts_real_pdf_and_deduplicates(self):
        first = self.upload(); self.assertEqual(first.status_code, 200, first.text)
        doc = first.json()['document']; self.assertEqual(doc['pages'], 2)
        self.assertEqual(self.upload().json()['document']['id'], doc['id'])
        self.assertEqual(len(rag.list_documents(self.workspace)), 1)
        hits = rag.retrieve(self.workspace, 'Ohm law resistance', [])
        self.assertTrue(hits); self.assertIn('Ohm', hits[0]['text']); self.assertEqual(hits[0]['page'], 1)

    def test_workspace_isolation_for_read_search_delete(self):
        doc = self.upload().json()['document']
        other = str(uuid.uuid4())
        self.assertEqual(rag.list_documents(other), [])
        self.assertEqual(rag.retrieve(other, 'electricity', []), [])
        self.assertFalse(rag.delete_document(other, doc['id']))
        response = self.client.post('/api/chat', json={'messages': [{'role': 'user', 'content': 'current'}], 'document_ids': [doc['id']]}, headers={**self.headers, 'X-Workspace-ID': other})
        self.assertEqual(response.status_code, 404)

    def test_delete_cascades_to_search(self):
        doc = self.upload().json()['document']
        self.assertEqual(self.client.delete(f"/api/documents/{doc['id']}", headers=self.headers).status_code, 200)
        self.assertEqual(rag.retrieve(self.workspace, 'current', []), [])
        self.assertEqual(rag.list_documents(self.workspace), [])
        self.assertEqual(self.client.delete(f"/api/documents/{doc['id']}", headers=self.headers).status_code, 404)

    def test_invalid_pdf_and_name_and_empty(self):
        self.assertEqual(self.upload(b'not a PDF').status_code, 415)
        self.assertEqual(self.upload(b'').status_code, 400)
        self.assertEqual(self.upload(name='notes.txt').status_code, 400)
        self.assertEqual(self.upload(name='a'*181 + '.pdf').status_code, 400)
        self.assertEqual(self.upload(name='bad\n.pdf').status_code, 400)
        self.assertEqual(self.upload(b'%PDF-broken').status_code, 422)
        result = self.upload(name='../../safe.pdf').json()
        self.assertEqual(result['document']['name'], 'safe.pdf')

    def test_scanned_and_encrypted_pdf(self):
        writer = PdfWriter(); writer.add_blank_page(width=200, height=200)
        body = BytesIO(); writer.write(body)
        self.assertEqual(self.upload(body.getvalue()).status_code, 422)
        writer.encrypt('password'); body = BytesIO(); writer.write(body)
        self.assertEqual(self.upload(body.getvalue()).status_code, 422)

    def test_upload_size_and_workspace_validation(self):
        self.assertEqual(self.upload(b'%PDF-' + b'x' * (20*1024*1024)).status_code, 413)
        self.assertEqual(self.upload(headers={'X-Workspace-ID': 'invalid'}).status_code, 400)
        self.assertEqual(self.client.get('/api/documents').status_code, 422)

    def test_document_quota(self):
        for i in range(20): rag.index_pages(self.workspace, f'{i}.pdf', ['current and voltage'], str(i))
        response = self.upload()
        self.assertEqual(response.status_code, 409)
        self.assertEqual(len(rag.list_documents(self.workspace)), 20)

    def test_document_scope(self):
        first = rag.index_pages(self.workspace, 'science.pdf', ['electric current and energy'], 'one')
        second = rag.index_pages(self.workspace, 'history.pdf', ['nationalism and revolution'], 'two')
        self.assertEqual(rag.retrieve(self.workspace, 'current', [second['id']]), [])
        self.assertEqual(rag.retrieve(self.workspace, 'current', [first['id']])[0]['name'], 'science.pdf')

    def test_summary_samples_beginning_and_end(self):
        rag.index_pages(self.workspace, 'long.pdf', [f'passage number {i}' for i in range(30)], 'long')
        result = rag.retrieve(self.workspace, 'Summarize this chapter in 6 bullet points.', [])
        self.assertEqual(len(result), 6)
        self.assertEqual(result[0]['page'], 1); self.assertEqual(result[-1]['page'], 30)

    def test_chat_requires_library_and_key(self):
        self.assertEqual(self.chat().status_code, 409)
        self.upload(); self.headers.pop('Authorization')
        self.assertEqual(self.chat().status_code, 401)

    def test_invalid_requests(self):
        self.assertEqual(self.chat(mode='bad').status_code, 422)
        self.assertEqual(self.chat(model='not-a-model').status_code, 422)
        self.assertEqual(self.chat(question='  ').status_code, 422)
        self.assertEqual(self.chat(document_ids=['bad']).status_code, 422)
        response = self.client.post('/api/chat', json={'messages':[{'role':'assistant', 'content':'hi'}]}, headers=self.headers)
        self.assertEqual(response.status_code, 400)
        response = self.client.post('/api/chat', json={'messages':[{'role':'user','content':'x'*12000}]*6}, headers=self.headers)
        self.assertEqual(response.status_code, 413)

    def test_byok_takes_precedence(self):
        with patch.dict(os.environ, {'NVIDIA_API_KEY': 'nvapi-server'}):
            self.assertEqual(rag.resolve_key(None), 'nvapi-server')
            self.assertEqual(rag.resolve_key('Bearer nvapi-browser'), 'nvapi-browser')
            with self.assertRaises(HTTPException): rag.resolve_key('Bearer invalid')
        for bad in ['Bearer nvapi- key', 'Basic xyz', 'Bearer ']:
            with self.assertRaises(HTTPException): rag.resolve_key(bad)

    @patch('api.rag_pipeline.requests.post')
    def test_cited_answer_passes_key_only_to_nvidia(self, post):
        self.upload(); post.return_value = Mock(ok=True, status_code=200)
        post.return_value.json.return_value = {'choices':[{'message':{'content':'Current is charge per unit time. [1]'}}]}
        response = self.chat(); self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()['sources'][0]['cited'])
        self.assertEqual(response.json()['sources'][0]['page'], 1)
        args, kwargs = post.call_args
        self.assertEqual(args[0], 'https://integrate.api.nvidia.com/v1/chat/completions')
        self.assertEqual(kwargs['headers']['Authorization'], f'Bearer {KEY}')
        self.assertEqual(kwargs['json']['model'], MODEL)
        self.assertNotIn(KEY, kwargs['json']['messages'][0]['content'])
        self.assertNotIn(KEY.encode(), (Path(self.temp.name)/'library.sqlite3').read_bytes())

    @patch('api.rag_pipeline.nim_request')
    def test_unknown_topic_does_not_call_provider(self, nim):
        self.upload(); response = self.chat(question='Explain quantum chromodynamics')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['sources'], []); nim.assert_not_called()

    @patch('api.rag_pipeline.nim_request')
    def test_follow_up_uses_previous_topic(self, nim):
        self.upload(); nim.return_value = {'choices':[{'message':{'content':'Current is measured in amperes. [1]'}}]}
        response = self.client.post('/api/chat', json={'messages':[{'role':'user','content':'What is electric current?'},{'role':'assistant','content':'Charge flow.'},{'role':'user','content':'Can you elaborate?'}]}, headers=self.headers)
        self.assertEqual(response.status_code, 200); self.assertTrue(response.json()['sources'])

    @patch('api.rag_pipeline.nim_request')
    def test_flashcards_are_validated_and_exportable(self, nim):
        self.upload(); nim.return_value = {'choices':[{'message':{'content':json.dumps({'cards':[{'question':'What is current?', 'answer':'Charge per unit time. [1]'}]})}}]}
        response = self.chat(mode='flashcards'); self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['cards'][0]['question'], 'What is current?')
        self.assertIn('Charge per unit time.', response.json()['answer'])
        nim.return_value = {'choices':[{'message':{'content':'{"cards":[{"question":"Q", "answer":"Unsupported [99]"}]}'}}]}
        self.assertEqual(self.chat(mode='flashcards').status_code, 502)

    @patch('api.rag_pipeline.requests.post')
    def test_provider_errors_are_actionable_and_redacted(self, post):
        self.upload()
        for code, expected in [(401,401),(403,401),(429,429),(404,400),(400,400),(422,400),(503,502)]:
            post.return_value = Mock(ok=False, status_code=code, text='secret provider error')
            response = self.chat(); self.assertEqual(response.status_code, expected)
            self.assertNotIn('secret provider error', response.text)
        post.side_effect = requests.Timeout()
        self.assertEqual(self.chat().status_code, 504)
        post.side_effect = requests.ConnectionError()
        self.assertEqual(self.chat().status_code, 502)

    @patch('api.rag_pipeline.requests.post')
    def test_unreadable_and_empty_provider_response(self, post):
        self.upload(); post.return_value = Mock(ok=True, status_code=200)
        for result in [[], {}, {'choices':[]}, {'choices':[{'message':{'content':None}}]}]:
            post.return_value.json.return_value = result
            self.assertEqual(self.chat().status_code, 502)
        post.return_value.json.side_effect = ValueError()
        self.assertEqual(self.chat().status_code, 502)

    @patch('api.rag_pipeline.nim_request')
    def test_connection_checks_actual_completion(self, nim):
        nim.return_value = {'choices':[{'message':{'content':'OK'}}]}
        response = self.client.post('/api/connection', json={'model':MODEL}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['connected'])
        nim.return_value = {'choices':[]}
        self.assertEqual(self.client.post('/api/connection', json={'model':MODEL}, headers=self.headers).status_code, 502)

if __name__ == '__main__': unittest.main()
