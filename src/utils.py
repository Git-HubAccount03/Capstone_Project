"""
Utility functions used across the project.
"""

import os


def get_pdf_files(document_folder):
    """
    Returns all PDF files inside the documents folder.
    """

    pdf_files = []

    if not os.path.exists(document_folder):
        return pdf_files

    for file in os.listdir(document_folder):

        if file.lower().endswith(".pdf"):
            pdf_files.append(os.path.join(document_folder, file))

    return pdf_files