"""Step 14 - standalone copy of the website for public hosting (GitHub Pages) -> site_public/

The hosted Artifact adds the page skeleton itself; a plain web host needs it in the file.
"""
import shutil

import _bootstrap  # noqa: F401
from solarpark import config

SRC, OUT = config.ROOT / "site", config.ROOT / "site_public"
HEAD = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<meta name="description" content="Hourly 1, 3, 6 and 24-hour solar power forecasts for India\'s MNRE-sanctioned solar parks, with model results.">\n'
        '<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
        'body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>\n</head>\n<body>\n')

body = (SRC / "index.html").read_text(encoding="utf-8")
assert "<!doctype" not in body.lower(), "site/index.html must stay skeleton-free (the Artifact host adds it)"
if OUT.exists():
    shutil.rmtree(OUT / "data", ignore_errors=True)
OUT.mkdir(exist_ok=True)
(OUT / "index.html").write_text(HEAD + body + "\n</body>\n</html>\n", encoding="utf-8")
shutil.copytree(SRC / "data", OUT / "data")
(OUT / ".nojekyll").write_text("")
(OUT / "README.md").write_text(
    "# Solar Park Forecast Console\n\n"
    "Static website for a B.Tech research project (NIT Patna, Department of Electrical Engineering): short-term solar power\n"
    "forecasts for India's MNRE-sanctioned solar parks. Test period January to September 2026.\n\n"
    "* 'Actual' output is simulated with pvlib from ERA5 weather; no park publishes hourly metered output.\n"
    "* Model inputs are ECMWF forecasts issued one or two days earlier (Open-Meteo).\n"
    "* Boundaries: datameet (Survey-of-India-based). Park positions approximate (OpenStreetMap).\n",
    encoding="utf-8")
print("built", OUT, sorted(p.name for p in OUT.iterdir()), sorted(p.name for p in (OUT / "data").iterdir()))
