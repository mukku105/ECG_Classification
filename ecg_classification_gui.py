import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class ECGClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clinical ECG Classifier")
        self.root.geometry("950x820")
        
        # Clinical thresholds (adjustable)
        self.abnormal_threshold = 0.7
        self.uncertain_range = 0.1
        
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
        # Header with clinical emphasis
        header = tk.Label(self.root, text="CLINICAL ECG CLASSIFICATION SYSTEM", 
                         font=("Arial", 16, "bold"), fg="#006400")
        header.pack(pady=10)
        
        # File Upload Section
        upload_frame = tk.LabelFrame(self.root, text="1. ECG Data Input", padx=10, pady=10)
        upload_frame.pack(pady=5, padx=20, fill="x")
        
        self.file_path = tk.StringVar()
        tk.Label(upload_frame, text="Select PTBDB CSV:").grid(row=0, column=0, sticky="w")
        tk.Entry(upload_frame, textvariable=self.file_path, width=50).grid(row=0, column=1)
        tk.Button(upload_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5)
        
        # Row Selection with clinical context
        row_frame = tk.LabelFrame(self.root, text="2. Sample Selection", padx=10, pady=10)
        row_frame.pack(pady=5, padx=20, fill="x")
        
        tk.Label(row_frame, text="Total Records:").grid(row=0, column=0, sticky="e")
        self.row_count_label = tk.Label(row_frame, text="0", width=10, anchor="w", fg="blue")
        self.row_count_label.grid(row=0, column=1, sticky="w")
        
        tk.Label(row_frame, text="Record #:").grid(row=0, column=2, padx=(10,0), sticky="e")
        self.row_entry = tk.Entry(row_frame, width=10)
        self.row_entry.grid(row=0, column=3, sticky="w")
        
        self.prev_btn = tk.Button(row_frame, text="◄ Previous", state="disabled", command=lambda: self.navigate(-1))
        self.prev_btn.grid(row=0, column=4, padx=(10,5))
        self.next_btn = tk.Button(row_frame, text="Next ►", state="disabled", command=lambda: self.navigate(1))
        self.next_btn.grid(row=0, column=5)
        
        analyze_btn = tk.Button(row_frame, text="Analyze ECG", command=self.analyze_ecg, 
                              bg="#006400", fg="white", font=("Arial", 10, "bold"))
        analyze_btn.grid(row=0, column=6, padx=(20,0))
        
        # Clinical Results Display
        results_frame = tk.LabelFrame(self.root, text="3. Clinical Interpretation", padx=10, pady=10)
        results_frame.pack(pady=5, padx=20, fill="both", expand=True)
        
        # Probability Display Frame
        prob_frame = tk.Frame(results_frame)
        prob_frame.pack(fill="x", pady=5)
        
        # Abnormal Probability (Red)
        tk.Label(prob_frame, text="Abnormal Probability:", font=("Arial", 11)).pack(side="left")
        self.abnormal_prob = tk.Label(prob_frame, text="0%", font=("Arial", 11, "bold"), fg="red")
        self.abnormal_prob.pack(side="left", padx=5)
        
        # Normal Probability (Green)
        tk.Label(prob_frame, text="Normal Probability:", font=("Arial", 11)).pack(side="left", padx=(20,0))
        self.normal_prob = tk.Label(prob_frame, text="0%", font=("Arial", 11, "bold"), fg="green")
        self.normal_prob.pack(side="left", padx=5)
        
        # Clinical Decision Frame
        decision_frame = tk.Frame(results_frame)
        decision_frame.pack(fill="x", pady=10)
        
        tk.Label(decision_frame, text="Clinical Decision:", font=("Arial", 12)).pack(side="left")
        self.decision_label = tk.Label(decision_frame, text="", font=("Arial", 12, "bold"))
        self.decision_label.pack(side="left", padx=10)
        
        # Confidence Indicator
        confidence_frame = tk.Frame(results_frame)
        confidence_frame.pack(fill="x")
        
        tk.Label(confidence_frame, text="Confidence Level:", font=("Arial", 11)).pack(side="left")
        self.confidence_bar = ttk.Progressbar(confidence_frame, orient="horizontal", length=200, mode="determinate")
        self.confidence_bar.pack(side="left", padx=5)
        self.confidence_text = tk.Label(confidence_frame, text="", font=("Arial", 11))
        self.confidence_text.pack(side="left")
        
        # ECG Visualization
        vis_frame = tk.Frame(results_frame)
        vis_frame.pack(fill="both", expand=True, pady=10)
        
        self.figure = plt.figure(figsize=(10, 4), dpi=100, facecolor="#f0f0f0")
        self.canvas = FigureCanvasTkAgg(self.figure, master=vis_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            try:
                self.df = pd.read_csv(filename, header=None)
                self.file_path.set(os.path.basename(filename))
                self.row_count_label.config(text=str(len(self.df)))
                self.current_row = 0
                self.row_entry.delete(0, tk.END)
                self.row_entry.insert(0, "0")
                self.prev_btn.config(state="normal")
                self.next_btn.config(state="normal")
                self.analyze_ecg()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
                
    def navigate(self, step):
        if self.df is None: return
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
            ecg_data = self.df.iloc[row_num, :-1].values.reshape(1, -1)
            X = self.scaler.transform(ecg_data)
            
            # Get prediction and probability
            prediction = self.model.predict(X, verbose=0)
            prob_abnormal = float(prediction[0][0])
            prob_normal = 1 - prob_abnormal
            
            # Update probability displays
            self.abnormal_prob.config(text=f"{prob_abnormal:.1%}")
            self.normal_prob.config(text=f"{prob_normal:.1%}")
            
            # Determine clinical decision
            if prob_abnormal > self.abnormal_threshold:
                decision = "ABNORMAL ECG"
                color = "red"
                confidence = prob_abnormal
            elif prob_abnormal > (0.5 - self.uncertain_range/2) and prob_abnormal < (0.5 + self.uncertain_range/2):
                decision = "UNCERTAIN - Physician Review Recommended"
                color = "orange"
                confidence = min(prob_abnormal, prob_normal) * 2
            else:
                decision = "NORMAL ECG"
                color = "green"
                confidence = prob_normal
            
            self.decision_label.config(text=decision, fg=color)
            
            # Update confidence indicator
            confidence_pct = confidence * 100
            self.confidence_bar["value"] = confidence_pct
            self.confidence_text.config(text=f"{confidence_pct:.1f}%")
            
            # Visualize ECG
            self.figure.clear()
            ax = self.figure.add_subplot(111, facecolor="#f0f0f0")
            
            plot_length = min(100, len(X[0])) if len(X[0]) > 200 else len(X[0])
            ax.plot(X[0, :plot_length], color='blue', linewidth=1.5)
            
            ax.set_title(f"ECG Signal - Record {row_num} (First {plot_length} Samples)", pad=20)
            ax.set_xlabel("Time (samples)", labelpad=10)
            ax.set_ylabel("Normalized Amplitude", labelpad=10)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Highlight if abnormal
            if prob_abnormal > self.abnormal_threshold:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
            
            self.canvas.draw()
            
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Analysis Error", f"ECG analysis failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ECGClassifierApp(root)
    root.mainloop()