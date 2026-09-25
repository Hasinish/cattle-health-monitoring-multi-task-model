import sys
import os
import re

sys.path.insert(0, os.path.abspath("."))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from scripts.sheet_tool import get_service, SPREADSHEET_ID, SHEET_IDS, robust_execute

def clean_latex_math_to_plain_text(text: str) -> str:
    """Transform raw LaTeX math, commands, and formatting into clean, natural English prose."""
    # 0. Table & Equation & Figure references
    text = re.sub(r'Tabletab:mtl_architecture_spec|Table~?\\ref\{tab:mtl_architecture_spec\}', 'Table 4.6', text)
    text = re.sub(r'Tabletab:mtl_training_protocol|Table~?\\ref\{tab:mtl_training_protocol\}', 'Table 4.7', text)
    text = re.sub(r'Tabletab:runstatus|Table~?\\ref\{tab:runstatus\}', 'Table 4.1', text)
    text = re.sub(r'Tabletab:datasets|Table~?\\ref\{tab:datasets\}', 'Table 4.2', text)
    text = re.sub(r'Tabletab:[a-zA-Z0-9_]+|Table~?\\ref\{[^}]+\}', 'the summary table', text)
    text = re.sub(r'Figurefig:phase3design|Figure~?\\ref\{fig:phase3design\}', 'Figure 4.1', text)
    text = re.sub(r'Figurefig:[a-zA-Z0-9_]+|Figure~?\\ref\{[^}]+\}', 'the system diagram', text)
    text = re.sub(r'Appendixapp:evidence|Appendix~?\\ref\{app:evidence\}', 'Appendix B', text)
    text = re.sub(r'Appendix~?\\ref\{[^}]+\}', 'Appendix B', text)
    text = re.sub(r'Equation\(eq:bcs_loss\)|Equation~?\\eqref\{eq:bcs_loss\}', 'the BCS loss formulation', text)
    text = re.sub(r'Equation\([a-zA-Z0-9_:]+\)|Equation~?\\eqref\{[^}]+\}', 'the loss formulation', text)
    text = re.sub(r'Chapter~?5', 'Chapter 5', text)

    # 1. Accents
    text = re.sub(r"Rodr\\'iguez|Rodr\'iguez|Rodr\\'\\?iguez", 'Rodriguez', text)
    text = re.sub(r"\\['`^\"~]([a-zA-Z])", r'\1', text)

    # 2. Mathematical structures & shapes
    text = re.sub(r'\\mathbf\{?H\}?\s*\\in\s*\\mathbb\{?R\}?\^\{?B_\{?\\mathrm\{?Beh\}?\}?\s*[×x*\\times]+\s*8\s*[×x*\\times]+\s*512\}?', 'H (shape [B_Beh, 8, 512])', text)
    text = re.sub(r'\\mathbf\{?H\}?\s*\\in\s*\\mathbb\{?R\}?\^B_Beh\s*[×x*\\times]+\s*8\s*[×x*\\times]+\s*512', 'H (shape [B_Beh, 8, 512])', text)
    text = re.sub(r'\\mathbf\{?H\}?_Beh\s*\\in\s*\\mathbb\{?R\}?\^\{?B_\{?\\mathrm\{?Beh\}?\}?\s*[×x*\\times]+\s*8\s*[×x*\\times]+\s*512\}?', 'H_Beh (shape [B_Beh, 8, 512])', text)
    text = re.sub(r'\\mathbf\{?H\}?_Beh\s*\\in\s*\\mathbb\{?R\}?\^B_Beh\s*[×x*\\times]+\s*8\s*[×x*\\times]+\s*512', 'H_Beh (shape [B_Beh, 8, 512])', text)
    text = re.sub(r'\\mathbf\{?H\}?\s*\\in\s*\\mathbb\{?R\}?\^B\s*[×x*\\times]+\s*8\s*[×x*\\times]+\s*512', 'H (shape [B, 8, 512])', text)
    text = re.sub(r'\\mathbf\{?h\}?_shared\s*\\in\s*\\mathbb\{?R\}?\^\{?512\}?', 'h_shared', text)
    text = re.sub(r'\\mathbf\{?h\}?\s*\\in\s*\\mathbb\{?R\}?\^\{?512\}?', 'h', text)
    text = re.sub(r'\\Delta\\mathbf\{?h\}?_t\s*\\in\s*\\mathbb\{?R\}?\^\{?512\}?', 'delta_h_t (512 dimensions)', text)
    text = re.sub(r'\\mathbb\{?R\}?\^\{?128\s*[×x*\\times]+\s*512\}?', 'dimensions [128, 512]', text)
    text = re.sub(r'\\mathbb\{?R\}?\^\{?512\s*[×x*\\times]+\s*128\}?', 'dimensions [512, 128]', text)
    text = re.sub(r'\\mathbb\{?R\}?\^\{?512\}?', '512 dimensions', text)
    text = re.sub(r'\\mathbb\{?R\}?\^\{?128\}?', '128 dimensions', text)
    text = re.sub(r'224\s*\\times\s*224|224\s*×\s*224', '224 x 224', text)

    # 3. Fractions, norms, cosine
    text = re.sub(r'\\cos\(\\mathbf\{?g\}?_i,\s*\\mathbf\{?g\}?_j\)\s*=\s*\\frac\\mathbf\{?g\}?_i\^\\top\s*\\mathbf\{?g\}?_j\\lVert\\mathbf\{?g\}?_i\\rVert_2\s*\\lVert\\mathbf\{?g\}?_j\\rVert_2', 'cos(g_i, g_j) = (g_i^T g_j) / (||g_i||_2 * ||g_j||_2)', text)
    text = re.sub(r'\\lVert\\mathbf\{?h\}?\\rVert_2', '||h||_2', text)
    text = re.sub(r'\\lVert\\mathbf\{?g\}?_i\\rVert_2', '||g_i||_2', text)
    text = re.sub(r'\\lVert\\mathbf\{?g\}?_j\\rVert_2', '||g_j||_2', text)
    text = re.sub(r'\\mathbb\{?I\}?\(\\mathbf\{?g\}?_i\^\\top\s*\\mathbf\{?g\}?_j\s*<\s*0\)', '(g_i^T g_j < 0)', text)

    # 4. Learning rates, constants, rounding
    text = re.sub(r'\\epsilon\s*=\s*10\^-8|\\epsilon\s*=\s*10\^\{-8\}', 'epsilon = 1e-8', text)
    text = re.sub(r'\\eta_0\s*=\s*10\^-4|\\eta_0\s*=\s*10\^\{-4\}', 'eta_0 = 1e-4', text)
    text = re.sub(r'\\eta_\\min\s*=\s*10\^-6|\\eta_\\min\s*=\s*10\^\{-6\}|\\eta_\{\\min\}\s*=\s*10\^\{-6\}', 'eta_min = 1e-6', text)
    text = re.sub(r'\\lceil\s*34,369\s*/\s*64\s*\\rceil', '34,369 / 64 (rounded up)', text)
    text = re.sub(r'10\^-4|10\^\{-4\}', '1e-4', text)
    text = re.sub(r'10\^-6|10\^\{-6\}', '1e-6', text)

    # 5. Model parameters and variables
    text = re.sub(r'\\mathbfg_shared\s*=\s*Σ_t\s*\\nabla_\\boldsymbol\\theta_shared\s*L_t', 'g_shared = sum_t grad(L_t)', text)
    text = re.sub(r'\\mathbfg_shared\s*=\s*Σ_t\s*\\mathbfg\'_t', "g_shared = sum_t g'_t", text)
    text = re.sub(r'\\boldsymbol\\theta_shared|\\boldsymbol\{\\theta\}_\{\\mathrm\{shared\}\}', 'theta_shared', text)
    text = re.sub(r'\\boldsymbol\\theta_head\^\(t\)|\\boldsymbol\{\\theta\}_\{\\mathrm\{head\}\}\^\{\(t\)\}', 'theta_head^(t)', text)
    text = re.sub(r'\\boldsymbol\\theta_adapter\^\(t\)|\\boldsymbol\{\\theta\}_\{\\mathrm\{adapter\}\}\^\{\(t\)\}', 'theta_adapter^(t)', text)
    text = re.sub(r'\\nabla_\\boldsymbol\\theta_shared\s*L_t', 'grad(L_t)', text)
    text = re.sub(r'\\nabla_theta_shared\s*L_t', 'grad(L_t)', text)
    text = re.sub(r'\\mathbf\{?g\}?\'_BCS', "g'_BCS", text)
    text = re.sub(r'\\mathbf\{?g\}?\'_Beh', "g'_Beh", text)
    text = re.sub(r'\\mathbf\{?g\}?\'_ReID', "g'_ReID", text)
    text = re.sub(r'\\mathbf\{?g\}?_i\^\\top\s*\\mathbf\{?g\}?_j', 'g_i^T g_j', text)
    text = re.sub(r'\\mathbf\{?g\}?_i', 'g_i', text)
    text = re.sub(r'\\mathbf\{?g\}?_j', 'g_j', text)
    text = re.sub(r'\\mathbf\{?g\}?_shared', 'g_shared', text)
    text = re.sub(r'\\mathbf\{?g\}?_BCS', 'g_BCS', text)
    text = re.sub(r'\\mathbf\{?g\}?_Beh', 'g_Beh', text)
    text = re.sub(r'\\mathbf\{?g\}?_ReID', 'g_ReID', text)

    text = re.sub(r'g_k\s*=\s*\\mathbf\{?w\}?_k\^\\top\s*\\mathbf\{?h\}?\s*\+\s*b_k', 'g_k = w_k^T h + b_k', text)
    text = re.sub(r'\\mathbf\{?w\}?_k\^\\top', 'w_k^T', text)
    text = re.sub(r'\\mathbf\{?W\}?_down\^\(t\)', 'W_down^(t)', text)
    text = re.sub(r'\\mathbf\{?b\}?_down\^\(t\)', 'b_down^(t)', text)
    text = re.sub(r'\\mathbf\{?W\}?_up\^\(t\)', 'W_up^(t)', text)
    text = re.sub(r'\\mathbf\{?b\}?_up\^\(t\)', 'b_up^(t)', text)
    text = re.sub(r'\\Delta\\mathbf\{?h\}?_t', 'delta_h_t', text)
    text = re.sub(r'\\mathbf\{?h\}?_shared', 'h_shared', text)
    text = re.sub(r'\\mathbf\{?h\}?_t', 'h_t', text)
    text = re.sub(r'\\mathbf\{?H\}?_Beh', 'H_Beh', text)
    text = re.sub(r'\\mathbf\{?H\}?', 'H', text)
    text = re.sub(r'\\mathbf\{?h\}?', 'h', text)
    text = re.sub(r'\\mathbf\{?z\}?\s*=\s*\\mathbf\{?h\}?\s*/\s*\\lVert\\mathbf\{?h\}?\\rVert_2', 'z = h / ||h||_2', text)
    text = re.sub(r'\\mathbf\{?z\}?', 'z', text)

    # 6. Losses, Batch sizes & Subscripts
    text = re.sub(r'L_BCS', '(L_BCS)', text)
    text = re.sub(r'L_Beh', '(L_Beh)', text)
    text = re.sub(r'L_ReID', '(L_ReID)', text)

    # 7. Sets, braces, and math symbols
    text = re.sub(r'\\in\s*\\0,\s*1,\s*2,\s*3,\s*4\\', 'in {0, 1, 2, 3, 4}', text)
    text = re.sub(r'\\in\s*\\0,\s*1,\s*2,\s*3\\', 'in {0, 1, 2, 3}', text)
    text = re.sub(r'\\in\s*\\0,\s*1\\', 'in {0, 1}', text)
    text = re.sub(r'\\in\s*\\BCS,\s*Beh,\s*ReID\\', 'in {BCS, Beh, ReID}', text)
    text = re.sub(r'\\in', 'in', text)
    text = re.sub(r'\\neq', '!=', text)
    text = re.sub(r'\\top', '^T', text)
    text = re.sub(r'\\ge|\\geq', '>=', text)
    text = re.sub(r'\\le|\\leq', '<=', text)
    text = re.sub(r'\\pm', '±', text)

    # 8. General cleanup of residual LaTeX wrappers
    text = re.sub(r'\\(mathbf|boldsymbol|mathrm|mathbb|mathcal|texttt|mathit)\{([^}]+)\}', r'\2', text)
    text = re.sub(r'\\(mathbf|boldsymbol|mathrm|mathbb|mathcal|texttt|mathit)([a-zA-Z])', r'\2', text)
    text = re.sub(r'\\([a-zA-Z]+)', r'\1', text)  # Any remaining escaped command name -> just name
    text = re.sub(r'\\', '', text)  # Any rogue stray backslashes
    text = re.sub(r'\(\(([^)]+)\)\)', r'(\1)', text)  # Double parens
    text = re.sub(r'\s{2,}', ' ', text)  # Extra spaces

    return text.strip()

def run_clean_prose(dry_run: bool = False):
    service = get_service()
    
    # 1. Fetch Chapter 4 and Chapter 2
    target_tabs = ["Chapter 2: Literature Review", "Chapter 4: Proposed Methodology"]
    ranges = [f"'{tab}'!A1:D320" for tab in target_tabs]
    res = robust_execute(lambda: service.values().batchGet(spreadsheetId=SPREADSHEET_ID, ranges=ranges).execute())
    
    updates = []
    
    for tab, val_range in zip(target_tabs, res.get("valueRanges", [])):
        rows = val_range.get("values", [])
        for idx, r in enumerate(rows, start=1):
            col_b = r[1] if len(r) > 1 else ""
            col_d = r[3] if len(r) > 3 else ""
            
            # Skip non-paraphrase formula rows
            if "LEAVE BLANK" in col_d or "🚫 DO NOT PARAPHRASE" in col_d:
                continue
                
            # Check if has LaTeX backslash or macros
            if "\\" in col_b or any(s in col_b for s in ["\\mathbf", "\\mathbb", "\\boldsymbol", "\\top", "\\nabla", "\\epsilon", "\\in", "\\times", "\\sigma", "\\theta", "\\sum", "\\lambda", "Tabletab:", "Equation("]):
                cleaned_b = clean_latex_math_to_plain_text(col_b)
                if "\\" in cleaned_b:
                    print(f"ERROR: Cleaning failed to eliminate all backslashes in {tab} Row {idx}!")
                    print(cleaned_b)
                    sys.exit(1)
                cell_ref = f"'{tab}'!B{idx}"
                updates.append({
                    "range": cell_ref,
                    "values": [[cleaned_b]],
                    "tab": tab,
                    "row": idx,
                    "old": col_b,
                    "new": cleaned_b
                })
                
    print(f"Found {len(updates)} cells requiring LaTeX cleanup across tabs.")
    for u in updates:
        print(f"\n[{u['tab']}] Row {u['row']} ({u['range']}):")
        print(f"  OLD: {u['old'][:100]}...")
        print(f"  NEW: {u['new'][:100]}...")
        
    if dry_run:
        print("\nDRY RUN complete. No changes made.")
        return
        
    # Execute batchUpdate on values
    body = {
        "valueInputOption": "RAW",
        "data": [{"range": u["range"], "values": u["values"]} for u in updates]
    }
    
    print("\nExecuting live batch update to Google Sheets...")
    result = robust_execute(lambda: service.values().batchUpdate(
        spreadsheetId=SPREADSHEET_ID,
        body=body
    ).execute())
    print(f"Success! Updated {result.get('totalUpdatedCells', 0)} cells across {len(result.get('responses', []))} ranges.")

if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    run_clean_prose(dry_run=dry)
