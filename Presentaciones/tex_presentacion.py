#!/usr/bin/env python3
import os
import argparse
import glob
import re
import subprocess
import tempfile
import shutil

def create_latex_comparison_slides(folder1, folder2, output_pdf):
    """
    Crea un PDF con diapositivas comparando imágenes de dos carpetas usando LaTeX/Beamer.
    
    Args:
        folder1: Ruta a la primera carpeta de imágenes
        folder2: Ruta a la segunda carpeta de imágenes
        output_pdf: Ruta del archivo PDF de salida
    """
    # Verificar que las carpetas existen
    if not os.path.isdir(folder1) or not os.path.isdir(folder2):
        raise ValueError("Ambas rutas deben ser directorios existentes")
    
    # Obtener lista de archivos de imagen en ambas carpetas
    ext_patterns = ["*.png", "*.jpg", "*.jpeg", "*.gif"]
    files1 = set()
    files2 = set()
    
    for pattern in ext_patterns:
        files1.update([os.path.basename(f) for f in glob.glob(os.path.join(folder1, pattern))])
        files2.update([os.path.basename(f) for f in glob.glob(os.path.join(folder2, pattern))])
    
    # Encontrar archivos comunes entre ambas carpetas
    common_files = sorted(list(files1.intersection(files2)))
    
    if not common_files:
        print(f"No se encontraron archivos de imagen comunes entre {folder1} y {folder2}")
        return
    
    # Obtener nombres más descriptivos de carpetas (dos niveles)
    def get_descriptive_path(path):
        parts = os.path.normpath(path).split(os.sep)
        if len(parts) >= 2:
            return f"{parts[-2]}/{parts[-1]}"
        else:
            return parts[-1]
    
    folder1_name = get_descriptive_path(folder1)
    folder2_name = get_descriptive_path(folder2)
    
    # Crear directorio temporal para archivos LaTeX y copiar imágenes
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copiar imágenes al directorio temporal
        for filename in common_files:
            src1 = os.path.join(folder1, filename)
            src2 = os.path.join(folder2, filename)
            
            # Asegurarse de que las imágenes existen
            if not os.path.exists(src1) or not os.path.exists(src2):
                print(f"Advertencia: No se puede encontrar {filename} en ambas carpetas. Saltando...")
                continue
            
            # Nombrar las imágenes como folder1_filename.ext y folder2_filename.ext
            base, ext = os.path.splitext(filename)
            dest1 = os.path.join(temp_dir, f"folder1_{base}{ext}")
            dest2 = os.path.join(temp_dir, f"folder2_{base}{ext}")
            
            shutil.copy2(src1, dest1)
            shutil.copy2(src2, dest2)
        
        # Generar el contenido del archivo LaTeX
        latex_content = []
        latex_content.append(r"\documentclass[aspectratio=169]{beamer}")
        latex_content.append(r"\usepackage[utf8]{inputenc}")
        latex_content.append(r"\usepackage[T1]{fontenc}")
        latex_content.append(r"\usepackage{graphicx}")
        latex_content.append(r"\usepackage{caption}")
        latex_content.append(r"\usepackage{subcaption}")
        latex_content.append(r"\usetheme{Madrid}")
        latex_content.append(r"\usecolortheme{beaver}")
        latex_content.append(r"\setbeamertemplate{navigation symbols}{}")  # Quitar símbolos de navegación
        latex_content.append(r"\setbeamertemplate{footline}[frame number]")  # Mostrar número de diapositiva
        
        # Título del documento
        latex_content.append(r"\title{Comparación de Imágenes}")
        latex_content.append(r"\subtitle{%s vs %s}" % (folder1_name, folder2_name))
        latex_content.append(r"\author{Generado automáticamente}")
        latex_content.append(r"\date{\today}")
        
        # Inicio del documento
        latex_content.append(r"\begin{document}")
        
        # Diapositiva por cada par de imágenes
        for filename in common_files:
            base, ext = os.path.splitext(filename)
            pretty_title = re.sub(r'[_-]', ' ', base).title()
            
            # Verificar que los archivos existen (después de copiarlos)
            img1_path = os.path.join(temp_dir, f"folder1_{base}{ext}")
            img2_path = os.path.join(temp_dir, f"folder2_{base}{ext}")
            
            if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                continue
            
            # Crear la diapositiva
            latex_content.append(r"\begin{frame}")
            latex_content.append(r"\frametitle{%s}" % pretty_title)
            
            # Añadir imágenes en dos columnas
            latex_content.append(r"\begin{figure}[ht]")
            latex_content.append(r"\centering")
            latex_content.append(r"\begin{subfigure}{0.48\textwidth}")
            latex_content.append(r"\centering")
            latex_content.append(r"\includegraphics[width=\textwidth]{%s}" % f"folder1_{base}{ext}")
            latex_content.append(r"\caption{%s}" % folder1_name)
            latex_content.append(r"\end{subfigure}")
            latex_content.append(r"\hfill")
            latex_content.append(r"\begin{subfigure}{0.48\textwidth}")
            latex_content.append(r"\centering")
            latex_content.append(r"\includegraphics[width=\textwidth]{%s}" % f"folder2_{base}{ext}")
            latex_content.append(r"\caption{%s}" % folder2_name)
            latex_content.append(r"\end{subfigure}")
            latex_content.append(r"\end{figure}")
            
            latex_content.append(r"\end{frame}")
        
        # Fin del documento
        latex_content.append(r"\end{document}")
        
        # Escribir el archivo LaTeX
        latex_file = os.path.join(temp_dir, "presentation.tex")
        with open(latex_file, "w") as f:
            f.write("\n".join(latex_content))
        
        # Compilar el archivo LaTeX a PDF
        try:
            # Primer paso: compilar con pdflatex
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "presentation.tex"],
                cwd=temp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            # Segundo paso: compilar de nuevo para referencias
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "presentation.tex"],
                cwd=temp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            # Copiar el PDF resultante al destino final
            pdf_file = os.path.join(temp_dir, "presentation.pdf")
            if os.path.exists(pdf_file):
                shutil.copy2(pdf_file, output_pdf)
                print(f"PDF creado exitosamente: {output_pdf}")
            else:
                print("Error: No se pudo generar el PDF")
        
        except subprocess.CalledProcessError as e:
            print(f"Error al compilar el archivo LaTeX: {e}")
            print("Salida de pdflatex:")
            print(e.stdout.decode('utf-8'))
            print(e.stderr.decode('utf-8'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crea un PDF con diapositivas comparando imágenes de dos carpetas usando LaTeX")
    parser.add_argument("folder1", help="Ruta a la primera carpeta de imágenes")
    parser.add_argument("folder2", help="Ruta a la segunda carpeta de imágenes")
    parser.add_argument("--output", "-o", default="comparacion_latex.pdf", 
                        help="Ruta del archivo PDF de salida (por defecto: comparacion_latex.pdf)")
    
    args = parser.parse_args()
    
    try:
        create_latex_comparison_slides(args.folder1, args.folder2, args.output)
    except Exception as e:
        print(f"Error: {e}")