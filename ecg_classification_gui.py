import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class ECGClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Classification System")
        self.root.geometry("900x700")
        
        # Load model and scaler
        try:
            model_path = os.path.join(os.getcwd(), "model", "ecg_model.keras")
            scaler_path = os.path.join(os.getcwd(), "model", "scaler.save")
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model/scaler: {str(e)}")
            self.root.destroy()
            return
        
        self.df = None
        self.current_row = 0
        self.create_widgets()
        
    def create_widgets(self):
        # Header
        header = tk.Label(self.root, text="ECG Classification System", 
                         font=("Arial", 16, "bold"))
        header.pack(pady=10)
        
        # File Upload Section
        upload_frame = tk.LabelFrame(self.root, text="1. Upload ECG Data", padx=10, pady=10)
        upload_frame.pack(pady=5, padx=20, fill="x")
        
        self.file_path = tk.StringVar()
        tk.Label(upload_frame, text="CSV File:").grid(row=0, column=0, sticky="w")
        tk.Entry(upload_frame, textvariable=self.file_path, width=50).grid(row=0, column=1)
        tk.Button(upload_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5)
        
        # Row Selection Section
        row_frame = tk.LabelFrame(self.root, text="2. Select ECG Sample", padx=10, pady=10)
        row_frame.pack(pady=5, padx=20, fill="x")
        
        tk.Label(row_frame, text="Total Rows:").grid(row=0, column=0, sticky="e")
        self.row_count_label = tk.Label(row_frame, text="0", width=10, anchor="w")
        self.row_count_label.grid(row=0, column=1, sticky="w")
        
        tk.Label(row_frame, text="Current Row:").grid(row=0, column=2, padx=(10,0), sticky="e")
        self.row_entry = tk.Entry(row_frame, width=10)
        self.row_entry.grid(row=0, column=3, sticky="w")
        
        self.prev_btn = tk.Button(row_frame, text="◄ Previous", state="disabled", command=lambda: self.navigate(-1))
        self.prev_btn.grid(row=0, column=4, padx=(10,5))
        self.next_btn = tk.Button(row_frame, text="Next ►", state="disabled", command=lambda: self.navigate(1))
        self.next_btn.grid(row=0, column=5)
        
        tk.Button(row_frame, text="Analyze", command=self.analyze_ecg, 
                 bg="#4CAF50", fg="white").grid(row=0, column=6, padx=(20,0))
        
        # Results Section
        results_frame = tk.LabelFrame(self.root, text="3. Results", padx=10, pady=10)
        results_frame.pack(pady=5, padx=20, fill="both", expand=True)
        
        self.result_text = tk.StringVar()
        self.result_text.set("Please load a CSV file and select a row to analyze")
        tk.Label(results_frame, textvariable=self.result_text, 
                font=("Arial", 12), wraplength=700).pack(pady=5)
        
        # ECG Plot
        self.figure = plt.figure(figsize=(10, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=results_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            try:
                self.df = pd.read_csv(filename, header=None)
                self.file_path.set(filename)
                self.row_count_label.config(text=str(len(self.df)))
                self.current_row = 0
                self.row_entry.delete(0, tk.END)
                self.row_entry.insert(0, "0")
                self.prev_btn.config(state="normal")
                self.next_btn.config(state="normal")
                self.analyze_ecg()  # Auto-analyze first row
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
                
    def navigate(self, step):
        if self.df is None:
            return
            
        new_row = max(0, min(len(self.df)-1, self.current_row + step))
        if new_row != self.current_row:
            self.current_row = new_row
            self.row_entry.delete(0, tk.END)
            self.row_entry.insert(0, str(self.current_row))
            self.analyze_ecg()
            
    def analyze_ecg(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please select a CSV file first")
            return
            
        try:
            row_num = int(self.row_entry.get())
            if row_num < 0 or row_num >= len(self.df):
                raise ValueError(f"Row must be between 0-{len(self.df)-1}")
                
            self.current_row = row_num
            
            # Get ECG data
            ecg_data = self.df.iloc[row_num, :-1].values.reshape(1, -1)
            X = self.scaler.transform(ecg_data)
            
            # Make prediction
            prediction = self.model.predict(X)
            probability = prediction[0][0]
            
            # Display results
            result = "Normal ECG" if probability < 0.5 else "Abnormal ECG"
            confidence = (1 - probability)*100 if probability < 0.5 else probability*100
            self.result_text.set(
                f"Row {row_num} Result: {result}\n"
                f"Confidence: {confidence:.2f}%\n"
                f"Probability: {probability:.4f}"
            )
            
            # Plot ECG signal
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Plot first 100 samples by default, full signal if < 200 samples
            plot_length = min(100, len(X[0])) if len(X[0]) > 200 else len(X[0])
            ax.plot(X[0, :plot_length], color='blue', linewidth=1.5)
            
            ax.set_title(f"ECG Signal - Row {row_num} (First {plot_length} Samples)")
            ax.set_xlabel("Time (samples)")
            ax.set_ylabel("Normalized Amplitude")
            ax.grid(True, linestyle='--', alpha=0.6)
            self.canvas.draw()
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze ECG: {str(e)}")
            
if __name__ == "__main__":
    root = tk.Tk()
    app = ECGClassifierApp(root)
    root.mainloop()