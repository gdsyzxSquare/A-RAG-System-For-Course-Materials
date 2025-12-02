"""
Advanced Data Cleaning Script for Course Documents
Specifically designed for PPT-extracted text with tables, images, and metadata
"""

import re
import os


class AdvancedDataCleaner:
    """Advanced cleaner for PPT-extracted course documents"""
    
    def __init__(self):
        self.cleaning_steps = []
    
    def clean(self, text: str) -> str:
        """
        Apply all cleaning steps
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        original_length = len(text)
        
        # Step 1: Remove HTML/Markdown tables
        text = self.remove_tables(text)
        
        # Step 2: Remove image references
        text = self.remove_images(text)
        
        # Step 3: Remove repeated page footers/headers
        text = self.remove_page_metadata(text)
        
        # Step 4: Clean special characters and symbols
        text = self.clean_special_chars(text)
        
        # Step 5: Normalize whitespace
        text = self.normalize_whitespace(text)
        
        # Step 6: Remove very short lines (likely artifacts)
        text = self.remove_short_lines(text)
        
        # Step 7: Restore proper paragraph structure
        text = self.restore_paragraphs(text)
        
        cleaned_length = len(text)
        reduction = (1 - cleaned_length / original_length) * 100
        
        print(f"✓ Cleaning complete:")
        print(f"  Original: {original_length:,} chars")
        print(f"  Cleaned: {cleaned_length:,} chars")
        print(f"  Reduction: {reduction:.1f}%")
        
        return text
    
    def remove_tables(self, text: str) -> str:
        """Remove HTML table tags and content"""
        # Remove complete table blocks
        text = re.sub(r'<table>.*?</table>', '', text, flags=re.DOTALL)
        return text
    
    def remove_images(self, text: str) -> str:
        """Remove markdown image references"""
        # Remove ![](images/...)
        text = re.sub(r'!\[.*?\]\(images/.*?\)', '', text)
        return text
    
    def remove_page_metadata(self, text: str) -> str:
        """Remove repeated page headers/footers"""
        # Remove lines like "Chengwei Qin (Al Thrust, Information Hub) | Introduction | 2025.09.04 | X/62"
        text = re.sub(r'Chengwei Qin.*?\d+/\d+', '', text)
        
        # Remove standalone page numbers
        text = re.sub(r'^\s*\d+\s*/\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        return text
    
    def clean_special_chars(self, text: str) -> str:
        """Clean unnecessary special characters"""
        # Replace bullet points with standard format
        text = text.replace('□', '- ')
        text = text.replace('■', '- ')
        text = text.replace('。', '. ')
        
        # Remove excessive dots/dashes used for spacing
        text = re.sub(r'\.{3,}', ' ', text)
        text = re.sub(r'-{3,}', '', text)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize all whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove trailing whitespace from lines
        text = re.sub(r' +$', '', text, flags=re.MULTILINE)
        
        return text
    
    def remove_short_lines(self, text: str, min_length: int = 3) -> str:
        """Remove very short lines that are likely artifacts"""
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Keep headers (starting with #) even if short
            if stripped.startswith('#') or len(stripped) >= min_length:
                filtered_lines.append(line)
            elif len(stripped) == 0:
                # Keep empty lines for paragraph breaks
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def restore_paragraphs(self, text: str) -> str:
        """Restore proper paragraph structure"""
        lines = text.split('\n')
        result = []
        current_paragraph = []
        
        for line in lines:
            stripped = line.strip()
            
            # Empty line = paragraph break
            if not stripped:
                if current_paragraph:
                    result.append(' '.join(current_paragraph))
                    result.append('')
                    current_paragraph = []
            # Header line (starts with #)
            elif stripped.startswith('#'):
                if current_paragraph:
                    result.append(' '.join(current_paragraph))
                    result.append('')
                    current_paragraph = []
                result.append(stripped)
                result.append('')
            # List item (starts with -, ., number)
            elif re.match(r'^[-\d.]', stripped):
                if current_paragraph:
                    result.append(' '.join(current_paragraph))
                    result.append('')
                    current_paragraph = []
                result.append(stripped)
            # Regular text line
            else:
                current_paragraph.append(stripped)
        
        # Add any remaining paragraph
        if current_paragraph:
            result.append(' '.join(current_paragraph))
        
        return '\n'.join(result)


def clean_course_documents(
    input_path: str = "data/raw/course_documents.txt",
    output_path: str = "data/raw/course_documents_cleaned.txt"
):
    """
    Clean course documents and save to new file
    
    Args:
        input_path: Path to raw document
        output_path: Path to save cleaned document
    """
    print(f"\n{'='*60}")
    print("CLEANING COURSE DOCUMENTS")
    print(f"{'='*60}\n")
    
    # Read raw text
    print(f"Reading from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    # Clean text
    cleaner = AdvancedDataCleaner()
    cleaned_text = cleaner.clean(raw_text)
    
    # Save cleaned text
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    print(f"\n✓ Cleaned text saved to: {output_path}")
    
    # Show preview
    print(f"\n{'='*60}")
    print("PREVIEW (first 1000 characters):")
    print(f"{'='*60}\n")
    print(cleaned_text[:1000])
    print("\n[...]")
    
    return cleaned_text


if __name__ == "__main__":
    cleaned_text = clean_course_documents()
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print("1. Review the cleaned text in: data/raw/course_documents_cleaned.txt")
    print("2. If satisfied, update config.yaml to use the cleaned file:")
    print("   data:")
    print("     raw_data_path: 'data/raw/course_documents_cleaned.txt'")
    print("3. Run: python main.py --mode build")
