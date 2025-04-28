import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ECGClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Classification System")
        self.root.geometry("800x600")
        
        # Load model and scaler (update paths if needed)
        try:
            self.model = load_model('/model/ecg_model.keras')
            self.scaler = joblib.load('/model/scaler.save')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model/scaler: {str(e)}")
            self.root.destroy()
            return
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Header
        header = tk.Label(self.root, text="ECG Classification System", 
                         font=("Arial", 16, "bold"))
        header.pack(pady=20)
        
        # File Upload Section
        upload_frame = tk.LabelFrame(self.root, text="Upload ECG Data", padx=10, pady=10)
        upload_frame.pack(pady=10, padx=20, fill="x")
        
        self.file_path = tk.StringVar()
        tk.Label(upload_frame, text="CSV File:").grid(row=0, column=0, sticky="w")
        tk.Entry(upload_frame, textvariable=self.file_path, width=50).grid(row=0, column=1)
        tk.Button(upload_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5)
        tk.Button(upload_frame, text="Analyze", command=self.analyze_ecg, 
                 bg="#4CAF50", fg="white").grid(row=1, column=1, pady=10)
        
        # Results Section
        results_frame = tk.LabelFrame(self.root, text="Results", padx=10, pady=10)
        results_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        self.result_text = tk.StringVar()
        self.result_text.set("Results will appear here")
        tk.Label(results_frame, textvariable=self.result_text, 
                font=("Arial", 12), wraplength=700).pack(pady=10)
        
        # ECG Plot
        self.figure = plt.figure(figsize=(10, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=results_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            self.file_path.set(filename)
            
    def analyze_ecg(self):
        filepath = self.file_path.get()
        if not filepath:
            messagebox.showwarning("Warning", "Please select a CSV file first")
            return
            
        try:
            # Load and preprocess data
            df = pd.read_csv(filepath, header=None)
            ecg_data = df.values
            
            if ecg_data.shape[1] != 188:  # Assuming PTBDB format (187 features + label)
                messagebox.showerror("Error", "Invalid data format. Expected 188 columns.")
                return
                
            # Separate features (assuming last column is label)
            X = ecg_data[:, :-1] if ecg_data.shape[1] == 188 else ecg_data
            X = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X)
            probability = prediction[0][0]
            
            # Display results
            result = "Normal ECG" if probability < 0.5 else "Abnormal ECG"
            confidence = f"({(1 - probability)*100:.2f}% confidence)" if probability < 0.5 else f"({probability*100:.2f}% confidence)"
            self.result_text.set(f"Result: {result}\nConfidence: {confidence}")
            
            # Plot ECG signal (first 100 samples for visualization)
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(X[0, :100], color='blue')
            ax.set_title("ECG Signal (First 100 Samples)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Amplitude (Normalized)")
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze ECG: {str(e)}")
            
if __name__ == "__main__":
    root = tk.Tk()
    app = ECGClassifierApp(root)
    root.mainloop()