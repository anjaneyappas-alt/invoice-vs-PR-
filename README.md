# Invoice vs PR Checker

Compares a supplier invoice against a Purchase Receive (PR), line by line,
and looks up SKU codes for anything the PR is missing.

## One-time setup

1. Install Python 3.10 or newer from python.org (tick "Add Python to PATH").
2. Open Command Prompt / Terminal in this folder and run:

       pip install -r requirements.txt

3. Get an API key from https://console.anthropic.com → API Keys → Create Key.
   Keep it safe; anyone with the key can spend on your account.

## Running it

    streamlit run app.py

The app opens in your browser at http://localhost:8501

## Using it

1. Paste your API key in the sidebar (or set ANTHROPIC_API_KEY once so you
   don't have to paste it every time).
2. Upload the Inventory Summary .xlsx in the sidebar.
3. Upload the invoice — photos of every page, or the PDF.
4. Upload the PR PDF. Its lines appear straight away; check them and fix any
   that read wrong before comparing.
5. Click "Check invoice against PR".
6. Download the result as CSV.

## Reading the result

- **match** — quantities agree
- **short** — invoice bills more than the PR received
- **excess** — PR received more than the invoice bills
- **not_in_pr** — the invoice has a line the PR does not (free items show here too)
- **not_in_invoice** — the PR has a line the invoice does not

Where a line has no PR SKU, the app looks the SKU up from the Inventory
Summary and writes the confidence in the Note column. Always check a
medium or low confidence SKU before posting it.

## Photo tips

Quantities get misread from bad photos. Shoot the bill flat and straight on,
in good light, with no shadow across the quantity column and the whole page
in frame. If a number looks wrong in the result, re-shoot that page.

## Cost

Roughly a few rupees per bill with Sonnet. Haiku is cheaper and Opus is more
accurate; switch in the sidebar. A long multi-page bill costs more than a
one-page bill.

## Limits

- Reads at most 8 pages per invoice.
- It reads what is printed. Always eyeball the invoice-lines table before
  acting on a difference, especially on handwritten or stamped bills.
