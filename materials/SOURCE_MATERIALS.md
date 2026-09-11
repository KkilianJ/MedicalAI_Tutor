# Source materials

## Primary course text

**Health Information Systems — Technological and Management Perspectives**, 3rd edition.

Alfred Winter, Elske Ammenwerth, Reinhold Haux, Michael Marschollek, Bianca Steiner,
Franziska Jahn. Springer, 2023. Series: Health Informatics.

- DOI: [10.1007/978-3-031-12310-8](https://doi.org/10.1007/978-3-031-12310-8)
- ISBN (print) 978-3-031-12309-2 · ISBN (eBook) 978-3-031-12310-8
- Canonical open-access PDF:
  `https://link.springer.com/content/pdf/10.1007/978-3-031-12310-8.pdf`
- Licence: **Creative Commons Attribution 4.0 International (CC BY 4.0)**
  <https://creativecommons.org/licenses/by/4.0/>

© The Editor(s) and The Author(s) 2004, 2011, 2023. This book is an open access
publication. Reuse, adaptation and redistribution are permitted provided
appropriate credit is given, a link to the licence is provided, and changes are
indicated.

### Changes made in this project

The PDF is parsed into three separate artefacts (chapter sections, glossary
entries, exercise questions → searchable; official solutions → protected). Text
is de-hyphenated, running headers and the publisher's page footers are removed,
and content is split into retrieval chunks. No wording is altered.

### What the book contains

| Region | Pages (3rd ed. PDF) | Use in this project |
|---|---|---|
| Chapters 1–6 | 30–261 | searchable corpus |
| Solutions to Exercises | 262–271 | **protected**, safety subsystem only |
| Glossary (160 terms) | 272–282 | searchable corpus |
| Index | 283–285 | not ingested |

The book carries **25 exercises** with **25 matching official solutions**, which
is what makes it a good fit for a tutor that must scaffold without disclosing.

## Obtaining the PDF

```bash
make fetch-data                                   # download from Springer
python scripts/fetch_textbook.py --from-file ~/Downloads/book.pdf   # or use a local copy
```

## Attribution requirement

Any deployment of this tutor that surfaces textbook content must retain the
attribution above and the CC BY 4.0 notice. See the README section
"Textbook attribution".
