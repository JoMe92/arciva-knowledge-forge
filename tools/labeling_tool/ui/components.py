
import streamlit.components.v1 as components

def inject_shortcuts():
    """
    Injects JavaScript to handle keyboard shortcuts.
    """
    js_code = """
    <script>
    document.addEventListener('keydown', function(e) {
        // Check for Ctrl (Windows/Linux) or Meta (Mac)
        const isCtrl = e.ctrlKey || e.metaKey;
        
        // Confirm & Next: Shift + Enter
        if (e.shiftKey && e.key === 'Enter') {
            const buttons = Array.from(document.getElementsByTagName('button'));
            const confirmBtn = buttons.find(b => b.innerText.includes('Confirm & Next'));
            
            if (confirmBtn) {
                e.preventDefault();
                confirmBtn.click();
            }
        }
        
        // Discard: Ctrl + Shift + Backspace
        if (isCtrl && e.shiftKey && (e.key === 'Backspace' || e.key === 'Delete')) {
             const buttons = Array.from(document.getElementsByTagName('button'));
             const discardBtn = buttons.find(b => b.innerText.includes('Discard Pair'));
             
             if (discardBtn) {
                 e.preventDefault();
                 discardBtn.click();
             }
        }
    });
    </script>
    """
    components.html(js_code, height=0, width=0)
