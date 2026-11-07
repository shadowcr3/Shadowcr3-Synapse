import re
import shutil
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FRONT = REPO / "frontend"
NODE = shutil.which("node")


class TestFrontendStatic(unittest.TestCase):
    def files(self, *suffixes):
        return [p for p in FRONT.iterdir() if p.suffix in suffixes]

    def test_no_inline_scripts_styles_or_handlers(self):
        html = (FRONT / "index.html").read_text()
        self.assertEqual(re.findall(r"<script(?![^>]*\bsrc=)[^>]*>", html), [], "inline <script>")
        self.assertNotIn("<style", html)
        self.assertIsNone(re.search(r'\sstyle\s*=', html), "inline style attribute")
        self.assertIsNone(re.search(r'\son[a-z]+\s*=', html), "inline event handler attribute")

    def test_every_referenced_asset_exists_and_is_served(self):
        html = (FRONT / "index.html").read_text()
        refs = re.findall(r'(?:src|href)="/static/([^"]+)"', html)
        self.assertGreaterEqual(len(refs), 8)
        for name in refs:
            self.assertTrue((FRONT / name).exists(), name)
        self.assertEqual(len(refs), len(set(refs)))

    def test_no_external_urls_or_unsafe_dom_apis(self):
        for p in self.files(".js", ".html", ".css"):
            src = re.sub(r"/\*.*?\*/", "", p.read_text(), flags=re.S)
            self.assertIsNone(re.search(r"https?://(?!www\.w3\.org)", src), f"{p.name}: external URL")
            self.assertIsNone(re.search(r"innerHTML|outerHTML|document\.write|\beval\(|new Function|insertAdjacentHTML", src), p.name)

    def test_no_browser_storage_or_cookies(self):
        for p in self.files(".js"):
            self.assertIsNone(re.search(r"localStorage|sessionStorage|document\.cookie|indexedDB", p.read_text()), p.name)

    def test_ids_used_by_scripts_exist_in_html(self):
        html = (FRONT / "index.html").read_text()
        ids = set(re.findall(r'\bid="([^"]+)"', html))
        used = set()
        for p in self.files(".js"):
            used |= set(re.findall(r"\$\('([a-z0-9-]+)'\)", p.read_text()))
        dynamic = {"tab-", "tabbtn-"}
        missing = {u for u in used if u not in ids and u + "-" not in dynamic and not any(u.startswith(d) for d in dynamic)}
        self.assertEqual(missing, set())

    def test_html_ids_are_unique(self):
        ids = re.findall(r'\bid="([^"]+)"', (FRONT / "index.html").read_text())
        self.assertEqual(len(ids), len(set(ids)))


@unittest.skipUnless(NODE, "node not installed")
class TestJs(unittest.TestCase):
    def test_js_syntax(self):
        for p in FRONT.glob("*.js"):
            r = subprocess.run([NODE, "--check", str(p)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{p.name}: {r.stderr}")

    def test_util_unit_tests(self):
        r = subprocess.run([NODE, str(REPO / "tests" / "js" / "util.test.js")], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


@unittest.skipUnless(NODE and (REPO / "tests" / "ui" / "node_modules" / "jsdom").exists(),
                     "UI test needs node and `cd tests/ui && npm install`")
class TestUiEndToEnd(unittest.TestCase):
    def test_ui_in_jsdom_against_live_server(self):
        r = subprocess.run([NODE, "ui.test.js"], cwd=REPO / "tests" / "ui", capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-1500:])


if __name__ == "__main__":
    unittest.main()
