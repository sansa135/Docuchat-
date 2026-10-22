import io
import json
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.main import app

client = TestClient(app)


def make_pdf(lines_by_page):
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for lines in lines_by_page:
        y = 800
        for line in lines:
            c.drawString(60, y, line)
            y -= 18
        c.showPage()
    c.save()
    return buf.getvalue()


def auth(email="a@test.com"):
    client.post("/auth/register", json={"email": email, "password": "password123"})
    tok = client.post("/auth/login", json={"email": email, "password": "password123"}).json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        ev, data = block.split("\n")
        events.append((ev.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def test_auth_required():
    assert client.get("/documents").status_code in (401, 403)


def test_bad_login():
    auth("b@test.com")
    r = client.post("/auth/login", json={"email": "b@test.com", "password": "wrongpass1"})
    assert r.status_code == 401


def test_upload_and_cited_answer():
    h = auth()
    pdf = make_pdf([["The refund policy allows returns within 30 days of purchase."],
                    ["Our office is located in Gurgaon near Cyber Hub."]])
    r = client.post("/documents", files={"file": ("policy.pdf", pdf, "application/pdf")}, headers=h)
    assert r.status_code == 201 and r.json()["chunks"] >= 2

    r = client.post("/chat", json={"question": "How many days to return a purchase?"}, headers=h)
    events = parse_sse(r.text)
    kinds = [e[0] for e in events]
    assert kinds[0] == "sources" and kinds[-1] == "done"
    top = events[0][1][0]
    assert top["filename"] == "policy.pdf" and top["page"] == 1


def test_cache_hit_on_repeat():
    h = auth()
    q = {"question": "Where is the office located?"}
    first = parse_sse(client.post("/chat", json=q, headers=h).text)
    second = parse_sse(client.post("/chat", json=q, headers=h).text)
    assert first[-1][1]["cached"] is False
    assert second[-1][1]["cached"] is True


def test_users_cannot_see_each_others_documents():
    h1, h2 = auth("u1@test.com"), auth("u2@test.com")
    pdf = make_pdf([["Secret launch code is ORCHID-77."]])
    doc = client.post("/documents", files={"file": ("secret.pdf", pdf, "application/pdf")}, headers=h1).json()
    events = parse_sse(client.post("/chat", json={"question": "launch code ORCHID"}, headers=h2).text)
    assert all(s["filename"] != "secret.pdf" for s in events[0][1])
    assert client.delete(f"/documents/{doc['id']}", headers=h2).status_code == 404


def test_rejects_non_pdf():
    h = auth()
    r = client.post("/documents", files={"file": ("a.txt", b"hi", "text/plain")}, headers=h)
    assert r.status_code == 400
