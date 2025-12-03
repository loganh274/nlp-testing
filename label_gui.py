import pandas as pd
import tkinter as tk
from tkinter import scrolledtext, messagebox

class LabelingApp:
    def __init__(self, input_file, output_file, text_column=2):
        self.df = pd.read_csv(input_file)
        self.texts = self.df.iloc[:, text_column].tolist()
        self.output_file = output_file
        self.results = []
        self.current_idx = 0
        
        self.labels = {
            0: "Very Negative",
            1: "Negative",
            2: "Neutral", 
            3: "Positive",
            4: "Very Positive"
        }
        
        self.setup_gui()
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Sentiment Labeling Tool")
        self.root.geometry("800x600")
        
        # Progress label
        self.progress_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.progress_label.pack(pady=5)
        
        # Text display area
        self.text_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=90, height=20, font=("Arial", 11))
        self.text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Button frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        # Label buttons with colors
        colors = ["#d32f2f", "#f57c00", "#9e9e9e", "#7cb342", "#388e3c"]
        for label_id, label_name in self.labels.items():
            btn = tk.Button(
                btn_frame, 
                text=f"{label_id}: {label_name}", 
                width=15, 
                bg=colors[label_id],
                fg="white",
                font=("Arial", 10, "bold"),
                command=lambda l=label_id: self.assign_label(l)
            )
            btn.pack(side=tk.LEFT, padx=5)
        
        # Skip and Save buttons
        ctrl_frame = tk.Frame(self.root)
        ctrl_frame.pack(pady=10)
        
        tk.Button(ctrl_frame, text="Skip (s)", width=10, command=self.skip).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl_frame, text="Save & Quit", width=10, command=self.save_and_quit).pack(side=tk.LEFT, padx=5)
        
        # Keyboard bindings
        self.root.bind('0', lambda e: self.assign_label(0))
        self.root.bind('1', lambda e: self.assign_label(1))
        self.root.bind('2', lambda e: self.assign_label(2))
        self.root.bind('3', lambda e: self.assign_label(3))
        self.root.bind('4', lambda e: self.assign_label(4))
        self.root.bind('s', lambda e: self.skip())
        self.root.bind('q', lambda e: self.save_and_quit())
        
        self.show_current_text()
        
    def show_current_text(self):
        if self.current_idx >= len(self.texts):
            self.save_and_quit()
            return
            
        self.progress_label.config(text=f"Progress: {self.current_idx + 1} / {len(self.texts)} | Labeled: {len(self.results)}")
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, self.texts[self.current_idx])
        
    def assign_label(self, label):
        self.results.append({
            'text': self.texts[self.current_idx],
            'label': label
        })
        self.current_idx += 1
        self.show_current_text()
        
    def skip(self):
        self.current_idx += 1
        self.show_current_text()
        
    def save_and_quit(self):
        if self.results:
            pd.DataFrame(self.results).to_csv(self.output_file, index=False)
            messagebox.showinfo("Saved", f"Saved {len(self.results)} labeled samples to {self.output_file}")
        self.root.quit()
        self.root.destroy()
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = LabelingApp('xpi_pull.csv', 'labeled_data.csv')
    app.run()