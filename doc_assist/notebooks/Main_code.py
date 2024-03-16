# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 00:47:37 2024

@author: UPPU HARISH
"""
import numpy as np
import pandas as pd
import openpyxl as pxl
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tkinter as tk
import sweetviz as sv
from tkinter import filedialog
import os
from PIL import Image, ImageTk

# UI Creation

class Doc_Assist:
    def __init__(self, master):
        self.master = master
        master.title("Doctor Assist")

        self.working_dir = ""
        self.excel_file = ""

        self.label_wd = tk.Label(master, text="Working Directory:")
        self.label_wd.grid(row=0, column=0)
        self.entry_wd = tk.Entry(master, width=50)
        self.entry_wd.grid(row=0, column=1)
        self.button_wd = tk.Button(master, text="Browse", command=self.browse_working_directory)
        self.button_wd.grid(row=0, column=2)

        self.label_ef = tk.Label(master, text="Dataset Excel File:")
        self.label_ef.grid(row=1, column=0)
        self.entry_ef = tk.Entry(master, width=50)
        self.entry_ef.grid(row=1, column=1)
        self.button_ef = tk.Button(master, text="Browse", command=self.browse_excel_file)
        self.button_ef.grid(row=1, column=2)

        self.label_params = []
        self.entry_params = []
        params = ["HAEMATOCRIT", "HAEMOGLOBINS", "ERYTHROCYTE", "LEUCOCYTE", "THROMBOCYTE", "MCH", "MCHC", "MCV", "AGE", "SEX (Enter Value 1 for MALE and 2 for FEMALE)"]
        for i, param in enumerate(params):
            label = tk.Label(master, text=param+":")
            label.grid(row=i+2, column=0)
            entry = tk.Entry(master)
            entry.grid(row=i+2, column=1)
            self.label_params.append(label)
            self.entry_params.append(entry)

        self.button_predict = tk.Button(master, text="Predict Treatment", command=self.predict)
        self.button_predict.grid(row=13, column=1)
        
        self.button_generatereport= tk.Button(master, text="Generate Feature Report", command=self.generate_feature_report)
        self.button_generatereport.grid(row=14, column=1)
        
        self.button_visualize= tk.Button(master, text="Generate more visualizations of Pateint Data", command=self.visualize_patient_data)
        self.button_visualize.grid(row=15, column=1)

    def browse_working_directory(self):
        self.working_dir = filedialog.askdirectory()
        self.entry_wd.delete(0, tk.END)
        self.entry_wd.insert(0, self.working_dir)

    def browse_excel_file(self):
        self.excel_file = filedialog.askopenfilename(filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*")))
        self.entry_ef.delete(0, tk.END)
        self.entry_ef.insert(0, self.excel_file)

    def predict(self):
        # Load data from Excel file
        if not self.excel_file:
            tk.messagebox.showerror("Error", "Please select an Excel file.")
            return
        try:
            df_raw = pd.read_excel(self.excel_file)
        except Exception as e:
            tk.messagebox.showerror("Error", f"Error reading Excel file: {e}")
            return
        # D. Machine Learning Model Development
        #import data from preprocessed source excel file
        df_m = df_raw
        df_m1 = df_m
        df_m1 = df_m1.replace("M",1)
        df_m1 = df_m1.replace("F",2)

        # Logistic Regression
        # Assuming target variable is in a column named 'SOURCE'
        X = df_m1.drop('SOURCE', axis=1)  # Features
        y = df_m1['SOURCE']  # Target variable

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the logistic regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        
        # Use trained model to predict outcomes
        new_data = []
        for entry in self.entry_params:
            new_data.append(float(entry.get()))

        prediction = model.predict([new_data])
        
        #References for Low and High Limits for CBC Parameters
        # Reference Data for MCH
        mch_l = 27
        mch_h = 31
        # Reference Data for MCHC
        mchc_l = 32
        mchc_h = 36
        # Reference Data for MCV
        mcv_l = 80
        mcv_h = 95
        # Reference Data for Haemoglobin
        hmg_m_l = 13.8
        hmg_m_h = 17.2
        hmg_f_l = 12.1
        hmg_f_h = 15.1
        # Reference Data for Hematocrit
        hmc_m_l = 40.7
        hmc_m_h = 50.3
        hmc_f_l = 36.1
        hmc_f_h = 44.3
        # Reference Data for Erythrocyte
        erth_m_l = 4.7
        erth_m_h = 6.1
        erth_f_l = 4.2
        erth_f_h = 5.4
        # Reference Data for Leucocyte
        leuc_l = 4.5
        leuc_h = 10
        # Reference Data for Thromobocyte
        thromb_l = 150
        thromb_h = 400   
        
        # Treatment predictions for different blood parameters
        if new_data[0] < hmc_m_l or new_data[1] < hmg_m_l or new_data[2] < erth_m_l:
            t = "Prescribe Oral iron supplements or intravenous iron, Vitamin B12 or folate, blood transfusion may be necessary to quickly increase hemoglobin levels"
        elif new_data[0] > hmc_m_h or new_data[1] > hmg_m_h or new_data[2] > erth_m_h:
            t = "Perform Phlebotomy (removal of blood) to reduce the number of red blood cells and decrease blood viscosity, improve oxygenation, medications such as hydroxyurea or interferon-alpha may be prescribed to reduce the production of red blood cells in polycythemia vera or in cases of secondary polycythemia"
        
        if new_data[5] < mch_l or new_data[6] < mchc_l or new_data[7] < mcv_l:
            t = "Prescribe Oral iron supplements or intravenous iron, blood transfusions and chelation therapy"
        elif new_data[5] > mch_h or new_data[6] > mchc_h or new_data[7] > mcv_h:
            t = "Prescribe Vitamin B12 supplementation, Folate supplementation"
        
        if new_data[3] < leuc_l:
            t = "Treat for Infections or diseases, Medication to stimulate WBC production"
        elif new_data[3] > leuc_h:
            t = "Prescribe Medications to reduce inflammation, Chemotherapy for Leukemia"
            
        if new_data[4] < thromb_l:
            t = "Prescribe Medications to boost Platelet production, Platelet transfusions in severe cases"
        elif new_data[4] > thromb_h:
            t = "Prescribe Medications to reduce platelet production, Blood thinners in certain cases to prevent clotting"
        
        # Display prediction
        tk.messagebox.showinfo("Prediction", f"The predicted outcome is: {prediction}")
        tk.messagebox.showinfo("Treatment Suggestion", f"Treatment suggested: {t}")

    def generate_feature_report(self):
        # Load data from Excel file
        if not self.excel_file:
            tk.messagebox.showerror("Error", "Please select an Excel file.")
            return
        try:
            df_raw = pd.read_excel(self.excel_file)
        except Exception as e:
            tk.messagebox.showerror("Error", f"Error reading Excel file: {e}")
            return
        df_m = df_raw
        # Generate a detail feature engineering data in HTML
        report = sv.analyze(df_m)
        report.show_html("feature_engineering_report.html")
        tk.messagebox.showinfo("feature_engineering_report.html", "Feature engineering report generated successfully! Please switch to your browser to view the report!")

    def visualize_patient_data(self):
        dir = "visuals"
        directory = os.path.join(self.working_dir,dir)
        # Load data from Excel file
        if not self.excel_file:
            tk.messagebox.showerror("Error", "Please select an Excel file.")
            return
        try:
            df_raw = pd.read_excel(self.excel_file)
        except Exception as e:
            tk.messagebox.showerror("Error", f"Error reading Excel file: {e}")
            return
        df_m = df_raw
        
        col_names = df_m.columns
        col_len = len(df_m[col_names[0]])
        
        # Visualization-1: Population of Males v Females
        plot_data_1 = df_m['SEX']
        value_counts = pd.Series(plot_data_1).value_counts().sort_index()    # Prepare data by getting counts of each value
        plt.figure()
        plt.bar(value_counts.index, value_counts.values)    # Create a bar chart
        
        for i, count in enumerate(value_counts):
            plt.text(i, count, str(count), ha='center', va='bottom')    # Add count labels on top of each bar
        
        plt.xlabel('Sex')    # Add x labels
        plt.ylabel('Count')    # Add y labels
        plt.title('Patient Population Distribution')     #Add title
        file1 = "Patient_Population_Distribution.png"
        plt.savefig(os.path.join(directory,file1))    # Save the plot as an image file
        
        # Visualization-2: Males vs Females and thier Risk Factor
        
        # Reference Data for MCH
        mch_l = 27
        mch_h = 31
        
        # Visualization-3: Males and thier MCH data
        
        # Scatter Distribution for Male MCH data
        mch_m = []
        
        for i in range(0,col_len):
            if df_m['SEX'][i] == "M":
                mch_m.append(df_m['MCH'][i])    # Prepare male MCH data
            #x_data_1.append(i)
        
        x_data_1 = np.arange(len(mch_m))
        mch_low = np.full(len(mch_m),mch_l)    #Prepare MCH lower limit data for Plot
        mch_high = np.full(len(mch_m),mch_h)   #Prepare MCH upper limit data for Plot
        
        plt.figure()
        plt.scatter(x_data_1, mch_m, label='Scatter Plot',s=2.5)
        plt.plot(x_data_1, mch_low, label='Line Plot', color='red')
        plt.plot(x_data_1, mch_high, label='Line Plot', color='green')
        
        plt.xlabel('Patient')
        plt.ylabel('MCH Value')
        plt.title('Distribution of MCH Value for Male Patients')
        file2 = "MCH Distirbution for Male Patients.png"
        plt.savefig(os.path.join(directory,file2))
        
        # Normal distribution and statistics for Male MCH data
        vert = np.arange(0,0.4,0.1)
        mch_low_1 = np.full(4,mch_l)
        mch_high_1 = np.full(4,mch_h)
        
        plt.figure()
        plt.hist(mch_m, bins=60, density=True, alpha=0.6, color='b')    # Plot histogram of the data
        mean = np.mean(mch_m)    # Calculate mean and standard deviation of the data
        std_dev = np.std(mch_m)    # Calculate mean and standard deviation of the data
        xmin = min(mch_m)
        xmax = max(mch_m)
        x = np.linspace(xmin, xmax, 100)    # Create a normal distribution plot
        p = norm.pdf(x, mean, std_dev)
        plt.plot(x, p, 'k', linewidth=2)
        plt.plot(mch_low_1, vert, label='MCH-Lower Limit', color='red')
        plt.plot(mch_high_1, vert, label='MCH-Upper Limit', color='green')
        title = "MCH Norm. Distribution - Male (Stats: Mean = %.2f,  SD = %.2f)" % (mean, std_dev)
        plt.title(title)
        plt.legend()
        file3 = "Normal Distribution for MCH-Male Patients.png"
        plt.savefig(os.path.join(directory,file3))
        
        # Visualization-4: Females and thier MCH data
        
        # Scatter Distribution for Female MCH data
        mch_f = []
        for i in range(0,col_len):
            if df_m['SEX'][i] == "F":
                mch_f.append(df_m['MCH'][i])    # Prepare male MCH data
            #x_data_2.append(i)
        
        x_data_2 = np.arange(len(mch_f))
        mch_low = np.full(len(mch_f),mch_l)    #Prepare MCH lower limit data for Plot
        mch_high = np.full(len(mch_f),mch_h)   #Prepare MCH upper limit data for Plot
        
        plt.figure()
        plt.scatter(x_data_2, mch_f, label='Scatter Plot',s=2.5)
        plt.plot(x_data_2, mch_low, label='MCH-Lower Limit', color='red')
        plt.plot(x_data_2, mch_high, label='MCH-Upper Limit', color='green')
        
        plt.xlabel('Patient')
        plt.ylabel('MCH Value')
        plt.title('Distribution of MCH Value for Female Patients')
        plt.legend()
        file4 = "MCH Distirbution for Female Patients.png"
        plt.savefig(os.path.join(directory,file4))
        
        # Normal distribution and statistics for Female MCH data
        vert = np.arange(0,0.4,0.1)
        mch_low_1 = np.full(4,mch_l)
        mch_high_1 = np.full(4,mch_h)
        
        plt.figure()
        plt.hist(mch_f, bins=60, density=True, alpha=0.6, color='b')    # Plot histogram of the data
        mean = np.mean(mch_f)    # Calculate mean and standard deviation of the data
        std_dev = np.std(mch_f)    # Calculate mean and standard deviation of the data
        xmin = min(mch_f)
        xmax = max(mch_f)
        x = np.linspace(xmin, xmax, 100)    # Create a normal distribution plot
        p = norm.pdf(x, mean, std_dev)
        plt.plot(x, p, 'k', linewidth=2)
        plt.plot(mch_low_1, vert, label='MCH-Lower Limit', color='red')
        plt.plot(mch_high_1, vert, label='MCH-Upper Limit', color='green')
        title = "MCH Norm. Distribution - Female (Stats: Mean = %.2f,  SD = %.2f)" % (mean, std_dev)
        plt.title(title)
        plt.legend()
        file5 = "Normal Distribution for MCH-Female Patients.png"
        plt.savefig(os.path.join(directory,file5))
        
        # Reference Data for MCHC
        mchc_l = 32
        mchc_h = 36
        
        # Visualization-5: Males and thier MCHC data
        
        # Scatter Distribution for Male MCHC data
        mchc_m = []
        
        for i in range(0,col_len):
            if df_m['SEX'][i] == "M":
                mchc_m.append(df_m['MCHC'][i])    # Prepare male MCH data
            #x_data_1.append(i)
        
        x_data_3 = np.arange(len(mchc_m))
        mchc_low = np.full(len(mchc_m),mchc_l)    #Prepare MCH lower limit data for Plot
        mchc_high = np.full(len(mchc_m),mchc_h)   #Prepare MCH upper limit data for Plot
        
        plt.figure()
        plt.scatter(x_data_3, mchc_m, label='Scatter Plot',s=2.5)
        plt.plot(x_data_3, mchc_low, label='MCHC-Lower Limit', color='red')
        plt.plot(x_data_3, mchc_high, label='MCHC-Upper Limit', color='green')
        
        plt.xlabel('Patient')
        plt.ylabel('MCHC Value')
        plt.title('Distribution of MCHC Value for Male Patients')
        plt.legend()
        file6 = "MCHC Distirbution for Male Patients.png"
        plt.savefig(os.path.join(directory,file6))
        
        # Normal distribution and statistics for Male MCHC data
        vert = np.arange(0,0.4,0.1)
        mchc_low_1 = np.full(4,mchc_l)
        mchc_high_1 = np.full(4,mchc_h)
        
        plt.figure()
        plt.hist(mchc_m, bins=60, density=True, alpha=0.6, color='b')    # Plot histogram of the data
        mean = np.mean(mchc_m)    # Calculate mean and standard deviation of the data
        std_dev = np.std(mchc_m)    # Calculate mean and standard deviation of the data
        xmin = min(mchc_m)
        xmax = max(mchc_m)
        x = np.linspace(xmin, xmax, 100)    # Create a normal distribution plot
        p = norm.pdf(x, mean, std_dev)
        plt.plot(x, p, 'k', linewidth=2)
        plt.plot(mchc_low_1, vert, label='MCHC-Lower Limit', color='red')
        plt.plot(mchc_high_1, vert, label='MCHC-Upper Limit', color='green')
        title = "MCHC Norm. Distribution - Male (Stats: Mean = %.2f,  SD = %.2f)" % (mean, std_dev)
        plt.title(title)
        plt.legend()
        file7 = "Normal Distribution for MCHC-Male Patients.png"
        plt.savefig(os.path.join(directory,file7))
        
        # Visualization-6: Females and thier MCHC data
        
        # Scatter Distribution for Female MCHC data
        mchc_f = []
        
        for i in range(0,col_len):
            if df_m['SEX'][i] == "F":
                mchc_f.append(df_m['MCHC'][i])    # Prepare male MCH data
            #x_data_1.append(i)
        
        x_data_4 = np.arange(len(mchc_f))
        mchc_low = np.full(len(mchc_f),mchc_l)    #Prepare MCH lower limit data for Plot
        mchc_high = np.full(len(mchc_f),mchc_h)   #Prepare MCH upper limit data for Plot
        
        plt.figure()
        plt.scatter(x_data_4, mchc_f, label='Scatter Plot',s=2.5)
        plt.plot(x_data_4, mchc_low, label='MCHC-Lower Limit', color='red')
        plt.plot(x_data_4, mchc_high, label='MCHC-Upper Limit', color='green')
        
        plt.xlabel('Patient')
        plt.ylabel('MCHC Value')
        plt.title('Distribution of MCHC Value for Female Patients')
        plt.legend()
        file8 = "MCHC Distirbution for Female Patients.png"
        plt.savefig(os.path.join(directory,file8))
        
        # Normal distribution and statistics for Female MCHC data
        vert = np.arange(0,0.4,0.1)
        mchc_low_1 = np.full(4,mchc_l)
        mchc_high_1 = np.full(4,mchc_h)
        
        plt.figure()
        plt.hist(mchc_f, bins=60, density=True, alpha=0.6, color='b')    # Plot histogram of the data
        mean = np.mean(mchc_f)    # Calculate mean and standard deviation of the data
        std_dev = np.std(mchc_f)    # Calculate mean and standard deviation of the data
        xmin = min(mchc_f)
        xmax = max(mchc_f)
        x = np.linspace(xmin, xmax, 100)    # Create a normal distribution plot
        p = norm.pdf(x, mean, std_dev)
        plt.plot(x, p, 'k', linewidth=2)
        plt.plot(mchc_low_1, vert, label='MCHC-Lower Limit', color='red')
        plt.plot(mchc_high_1, vert, label='MCHC-Upper Limit', color='green')
        title = "MCHC Norm. Distribution - Female (Stats: Mean = %.2f,  SD = %.2f)" % (mean, std_dev)
        plt.title(title)
        plt.legend()
        file9 = "Normal Distribution for MCHC-Female Patients.png"
        plt.savefig(os.path.join(directory,file9))
        
        # Reference Data for MCV
        mcv_l = 80
        mcv_h = 95
        
        # Visualization-7: Males and thier MCV data
        
        # Scatter Distribution for Male MCV data
        mcv_m = []
        
        for i in range(0,col_len):
            if df_m['SEX'][i] == "M":
                mcv_m.append(df_m['MCV'][i])    # Prepare male MCH data
            #x_data_1.append(i)
        
        x_data_5 = np.arange(len(mcv_m))
        mcv_low = np.full(len(mcv_m),mcv_l)    #Prepare MCH lower limit data for Plot
        mcv_high = np.full(len(mcv_m),mcv_h)   #Prepare MCH upper limit data for Plot
        
        plt.figure()
        plt.scatter(x_data_5, mcv_m, label='Scatter Plot',s=2.5)
        plt.plot(x_data_5, mcv_low, label='MCV-Lower Limit', color='red')
        plt.plot(x_data_5, mcv_high, label='MCV-Upper Limit', color='green')
        
        plt.xlabel('Patient')
        plt.ylabel('MCV Value')
        plt.title('Distribution of MCV Value for Male Patients')
        plt.legend()
        file10 = "MCV Distirbution for Male Patients.png"
        plt.savefig(os.path.join(directory,file10))
        
        # Normal distribution and statistics for Male MCV data
        vert = np.arange(0,0.1,0.05)
        mcv_low_1 = np.full(2,mcv_l)
        mcv_high_1 = np.full(2,mcv_h)
        
        plt.figure()
        plt.hist(mcv_m, bins=60, density=True, alpha=0.6, color='b')    # Plot histogram of the data
        mean = np.mean(mcv_m)    # Calculate mean and standard deviation of the data
        std_dev = np.std(mcv_m)    # Calculate mean and standard deviation of the data
        xmin = min(mcv_m)
        xmax = max(mcv_m)
        x = np.linspace(xmin, xmax, 100)    # Create a normal distribution plot
        p = norm.pdf(x, mean, std_dev)
        plt.plot(x, p, 'k', linewidth=2)
        plt.plot(mcv_low_1, vert, label='MCV-Lower Limit', color='red')
        plt.plot(mcv_high_1, vert, label='MCV-Upper Limit', color='green')
        title = "MCV Norm. Distribution - Male (Stats: Mean = %.2f,  SD = %.2f)" % (mean, std_dev)
        plt.title(title)
        plt.legend()
        file11 = "Normal Distribution for MCV-Male Patients.png"
        plt.savefig(os.path.join(directory,file11))
        
        # Visualization-8: Females and thier MCV data
        mcv_f = []
        
        for i in range(0,col_len):
            if df_m['SEX'][i] == "F":
                mcv_f.append(df_m['MCV'][i])    # Prepare male MCH data
            #x_data_1.append(i)
        
        x_data_6 = np.arange(len(mcv_f))
        mcv_low = np.full(len(mcv_f),mcv_l)    #Prepare MCH lower limit data for Plot
        mcv_high = np.full(len(mcv_f),mcv_h)   #Prepare MCH upper limit data for Plot
        
        plt.figure()
        plt.scatter(x_data_6, mcv_f, label='Scatter Plot',s=2.5)
        plt.plot(x_data_6, mcv_low, label='MCV-Lower Limit', color='red')
        plt.plot(x_data_6, mcv_high, label='MCV-Upper Limit', color='green')
        
        plt.xlabel('Patient')
        plt.ylabel('MCV Value')
        plt.title('Distribution of MCV Value for Female Patients')
        plt.legend()
        plt.savefig(r'C:\Users\UPPU HARISH\Desktop\divya\Capstone Project\doc_assist\visuals\MCV Distirbution for Female Patients.png')    # Save the plot as an image file
        
        # Normal distribution and statistics for Female MCV data
        vert = np.arange(0,0.1,0.05)
        mcv_low_1 = np.full(2,mcv_l)
        mcv_high_1 = np.full(2,mcv_h)
        
        plt.figure()
        plt.hist(mcv_f, bins=60, density=True, alpha=0.6, color='b')    # Plot histogram of the data
        mean = np.mean(mcv_f)    # Calculate mean and standard deviation of the data
        std_dev = np.std(mcv_f)    # Calculate mean and standard deviation of the data
        xmin = min(mcv_f)
        xmax = max(mcv_f)
        x = np.linspace(xmin, xmax, 100)    # Create a normal distribution plot
        p = norm.pdf(x, mean, std_dev)
        plt.plot(x, p, 'k', linewidth=2)
        plt.plot(mcv_low_1, vert, label='MCV-Lower Limit', color='red')
        plt.plot(mcv_high_1, vert, label='MCV-Upper Limit', color='green')
        title = "MCV Norm. Distribution - Female (Stats: Mean = %.2f,  SD = %.2f)" % (mean, std_dev)
        plt.title(title)
        plt.legend()
        file12 = "Normal Distribution for MCV-Female Patients.png"
        plt.savefig(os.path.join(directory,file12))
        
        tk.messagebox.showinfo("Patient Data Visualizations","Visualizations are generated! Please refer Visuals folder!")

def main():
    root = tk.Tk()
    app = Doc_Assist(root)
    root.mainloop()
if __name__ == "__main__":
    main()
