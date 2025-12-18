import csv
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Sentiment scale reference
SENTIMENT_SCALE = {
    0: ("Very Negative", "Angry, threatening to leave, harsh criticism", "#d32f2f"),
    1: ("Negative", "Frustrated, disappointed, complaining", "#f57c00"),
    2: ("Neutral", "Feature request, question, factual statement", "#9e9e9e"),
    3: ("Positive", "Grateful, satisfied, mild appreciation", "#7cb342"),
    4: ("Very Positive", "Enthusiastic, highly satisfied, strong praise", "#2e7d32")
}


class LabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Comment Labeling Tool")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        
        # Data
        self.labeled_data = []
        self.unlabeled_comments = []
        self.current_index = 0
        self.input_file = None
        self.output_file = None
        self.session_count = 0
        
        self.setup_ui()
        self.bind_shortcuts()
    
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="Files", padding="5")
        file_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Button(file_frame, text="Load Comments", command=self.load_input_file).grid(row=0, column=0, padx=5)
        self.input_label = ttk.Label(file_frame, text="No file loaded")
        self.input_label.grid(row=0, column=1, sticky="w", padx=5)
        
        ttk.Button(file_frame, text="Set Output File", command=self.set_output_file).grid(row=1, column=0, padx=5, pady=5)
        self.output_label = ttk.Label(file_frame, text="No output file set")
        self.output_label.grid(row=1, column=1, sticky="w", padx=5)
        
        # Progress frame
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_label = ttk.Label(progress_frame, text="Progress: 0 / 0")
        self.progress_label.grid(row=0, column=0, sticky="w")
        
        self.session_label = ttk.Label(progress_frame, text="Session: 0 labeled")
        self.session_label.grid(row=0, column=1, sticky="e")
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        
        # Comment display
        comment_frame = ttk.LabelFrame(main_frame, text="Comment", padding="10")
        comment_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        comment_frame.columnconfigure(0, weight=1)
        comment_frame.rowconfigure(0, weight=1)
        
        self.comment_text = tk.Text(comment_frame, wrap=tk.WORD, font=("Arial", 12), 
                                     state=tk.DISABLED, bg="#f5f5f5")
        self.comment_text.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = ttk.Scrollbar(comment_frame, orient=tk.VERTICAL, command=self.comment_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.comment_text.configure(yscrollcommand=scrollbar.set)
        
        # Label buttons frame
        button_frame = ttk.LabelFrame(main_frame, text="Label (keyboard shortcuts: 0-4, S=skip, ←/→ navigate)", padding="10")
        button_frame.grid(row=3, column=0, sticky="ew")
        button_frame.columnconfigure((0,1,2,3,4), weight=1)
        
        self.label_buttons = []
        for i, (label, (name, desc, color)) in enumerate(SENTIMENT_SCALE.items()):
            btn = tk.Button(button_frame, text=f"{label}: {name}", 
                           command=lambda l=label: self.apply_label(l),
                           bg=color, fg="white", font=("Arial", 10, "bold"),
                           activebackground=color, activeforeground="white")
            btn.grid(row=0, column=i, padx=3, pady=5, sticky="ew")
            self.label_buttons.append(btn)
            
            # Tooltip
            tooltip = ttk.Label(button_frame, text=desc, font=("Arial", 8))
            tooltip.grid(row=1, column=i, padx=3)
        
        # Navigation frame
        nav_frame = ttk.Frame(main_frame)
        nav_frame.grid(row=4, column=0, sticky="ew", pady=10)
        nav_frame.columnconfigure(1, weight=1)
        
        ttk.Button(nav_frame, text="← Previous", command=self.prev_comment).grid(row=0, column=0, padx=5)
        ttk.Button(nav_frame, text="Skip", command=self.skip_comment).grid(row=0, column=1)
        ttk.Button(nav_frame, text="Next →", command=self.next_comment).grid(row=0, column=2, padx=5)
        ttk.Button(nav_frame, text="Save & Exit", command=self.save_and_exit).grid(row=0, column=3, padx=20)
    
    def bind_shortcuts(self):
        self.root.bind('0', lambda e: self.apply_label(0))
        self.root.bind('1', lambda e: self.apply_label(1))
        self.root.bind('2', lambda e: self.apply_label(2))
        self.root.bind('3', lambda e: self.apply_label(3))
        self.root.bind('4', lambda e: self.apply_label(4))
        self.root.bind('s', lambda e: self.skip_comment())
        self.root.bind('S', lambda e: self.skip_comment())
        self.root.bind('<Left>', lambda e: self.prev_comment())
        self.root.bind('<Right>', lambda e: self.next_comment())
        self.root.bind('<Control-s>', lambda e: self.save_data())
        self.root.bind('<Command-s>', lambda e: self.save_data())  # Mac
    
    def load_input_file(self):
        filepath = filedialog.askopenfilename(
            title="Select comments file",
            filetypes=[("All supported", "*.csv *.txt"), ("CSV files", "*.csv"), ("Text files", "*.txt")]
        )
        if filepath:
            self.input_file = filepath
            self.input_label.config(text=os.path.basename(filepath))
            self.load_comments()
            
            # Auto-set output file
            if not self.output_file:
                base = os.path.splitext(filepath)[0]
                self.output_file = f"{base}_labeled.csv"
                self.output_label.config(text=os.path.basename(self.output_file))
                self.load_existing_labels()
    
    def set_output_file(self):
        filepath = filedialog.asksaveasfilename(
            title="Set output file",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if filepath:
            self.output_file = filepath
            self.output_label.config(text=os.path.basename(filepath))
            self.load_existing_labels()
    
    def load_comments(self):
        if not self.input_file:
            return
        
        comments = []
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                f.seek(0)
                
                if ',' in first_line and ('text' in first_line.lower() or 'comment' in first_line.lower()):
                    reader = csv.DictReader(f)
                    for row in reader:
                        text = row.get('text') or row.get('comment') or row.get('content') or list(row.values())[0]
                        if text:
                            comments.append(text.strip())
                else:
                    for line in f:
                        line = line.strip()
                        if line:
                            comments.append(line)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            return
        
        self.unlabeled_comments = comments
        self.filter_unlabeled()
        self.update_display()
    
    def load_existing_labels(self):
        if self.output_file and os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    self.labeled_data = list(reader)
            except Exception:
                self.labeled_data = []
        self.filter_unlabeled()
        self.update_display()
    
    def filter_unlabeled(self):
        labeled_texts = {item['text'] for item in self.labeled_data}
        self.unlabeled_comments = [c for c in self.unlabeled_comments if c not in labeled_texts]
        self.current_index = 0
    
    def update_display(self):
        total = len(self.unlabeled_comments)
        
        # Update progress
        self.progress_label.config(text=f"Progress: {self.current_index + 1} / {total}" if total > 0 else "Progress: 0 / 0")
        self.session_label.config(text=f"Session: {self.session_count} labeled | Total: {len(self.labeled_data)}")
        self.progress_bar['value'] = ((self.current_index + 1) / total * 100) if total > 0 else 0
        
        # Update comment display
        self.comment_text.config(state=tk.NORMAL)
        self.comment_text.delete(1.0, tk.END)
        if total > 0 and self.current_index < total:
            self.comment_text.insert(tk.END, self.unlabeled_comments[self.current_index])
        else:
            self.comment_text.insert(tk.END, "No comments to label. Load a file to begin.")
        self.comment_text.config(state=tk.DISABLED)
    
    def apply_label(self, label):
        if not self.unlabeled_comments or self.current_index >= len(self.unlabeled_comments):
            return
        
        comment = self.unlabeled_comments[self.current_index]
        self.labeled_data.append({'text': comment, 'label': label})
        self.session_count += 1
        
        # Auto-save every 10 labels
        if self.session_count % 10 == 0:
            self.save_data()
        
        # Move to next
        self.unlabeled_comments.pop(self.current_index)
        if self.current_index >= len(self.unlabeled_comments):
            self.current_index = max(0, len(self.unlabeled_comments) - 1)
        
        self.update_display()
        
        if not self.unlabeled_comments:
            self.save_data()
            messagebox.showinfo("Done", f"All comments labeled!\nTotal: {len(self.labeled_data)}")
    
    def skip_comment(self):
        if self.unlabeled_comments:
            self.current_index = (self.current_index + 1) % len(self.unlabeled_comments)
            self.update_display()
    
    def next_comment(self):
        if self.unlabeled_comments:
            self.current_index = (self.current_index + 1) % len(self.unlabeled_comments)
            self.update_display()
    
    def prev_comment(self):
        if self.unlabeled_comments:
            self.current_index = (self.current_index - 1) % len(self.unlabeled_comments)
            self.update_display()
    
    def save_data(self):
        if not self.output_file:
            self.set_output_file()
        if not self.output_file:
            return
        
        try:
            with open(self.output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['text', 'label'])
                writer.writeheader()
                writer.writerows(self.labeled_data)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")
    
    def save_and_exit(self):
        self.save_data()
        messagebox.showinfo("Saved", f"Saved {len(self.labeled_data)} labels to:\n{self.output_file}")
        self.root.quit()


def main():
    root = tk.Tk()
    app = LabelingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()