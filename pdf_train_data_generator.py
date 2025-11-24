"""
PDF Training Data Generator

Features:
- Traverse all PDF files in the specified directory
- Extract text content from PDF files
- Use Alibaba Cloud DashScope large model to generate training data
- Save as JSON format

Usage Instructions:
1. Install dependencies: pip install -r requirements.txt
2. Run the script: python pdf_train_data_generator.py
3. Follow the prompts to enter parameters

Note:
- The script uses Alibaba Cloud DashScope's OpenAI compatible mode API by default
- Supports Qwen series models (qwen-plus, qwen-max, qwen-turbo, etc.)
- Requires a valid Alibaba Cloud DashScope API key
"""

import json
import re
import asyncio
from datetime import datetime
import os
import PyPDF2
import pdfplumber
from openai import OpenAI, APIError

def extract_text_from_pdf(pdf_path):
    """Extract text content from PDF file"""
    text_content = ""
    metadata = {}
    file_name = os.path.basename(pdf_path)
    
    try:
        # Prefer using pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                metadata = pdf.metadata or {}
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n\n"
        except Exception as e:
            print(f"pdfplumber failed to process file {pdf_path}: {str(e)}, trying PyPDF2...")
            
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    metadata = pdf_reader.metadata or {}
                    
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text_content += page.extract_text() + "\n\n"
            except Exception as e2:
                raise Exception(f"Both PDF libraries failed to process: {str(e2)}")
    
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")
    
    return {
        'url': f"file://{pdf_path}",
        'title': metadata.get('Title', file_name) if isinstance(metadata.get('Title'), str) else file_name,
        'content': text_content,
        'metadata': {
            'source': 'pdf',
            'file_path': pdf_path,
            'file_name': file_name,
            **{k: v for k, v in metadata.items() if isinstance(v, str)}
        }
    }

class LLMClient:
    """LLM Client Wrapper
    
    Note: Uses Alibaba Cloud DashScope's OpenAI compatible mode by default,
    requires Alibaba Cloud DashScope API key and supported model name
    """
    
    def __init__(self, provider, api_key, base_url=None):
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.client = None
    
    def init_client(self):
        """Initialize LLM client"""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url.strip() if self.base_url else None
        )
    
    async def call_llm(self, messages, extra_params=None):
        """Call LLM API"""
        try:
            self.client = self.init_client()
            # Parse extra parameters
            extra_body = {}
            if extra_params:
                try:
                    extra_body = json.loads(extra_params)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse extra parameters: {e}")
            # Determine model name
            model_name = self.provider.split('/')[-1] if '/' in self.provider else self.provider
            # Call API
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
                extra_body=extra_body
            )
            return completion.choices[0].message.content
        except APIError as e:
            error_msg = str(e)
            # Remove error information that may contain API key
            if "api_key" in error_msg.lower():
                error_msg = "API call failed, please check API key and network connection"
            raise Exception(f"LLM API error: {error_msg}")
        except Exception as e:
            raise Exception(f"LLM API call failed: {str(e)}")
    
    def fix_json_response(self, response):
        """Attempt to fix invalid JSON response"""
        try:
            # Remove comments
            response = re.sub(r'//.*?\n|/\*.*?\*/', '', response, flags=re.DOTALL)
            # Try to parse
            return json.loads(response)
        except:
            try:
                # Try to extract the first valid JSON object
                json_str = re.search(r'\[[\s\S]*\]|\{[\s\S]*\}', response)
                if json_str:
                    return json.loads(json_str.group(0))
            except:
                return None
        return None

def generate_training_data(contents, provider, api_key, base_url, system_prompt, prompt_template, extra_params=None):
    """Generate training data"""
    llm_client = LLMClient(provider, api_key, base_url)
    results = []
    
    async def process_all():
        total = len(contents)
        
        for i, item in enumerate(contents):
            print(f"Processing progress: {i+1}/{total} - {item['title'] or item['url']}")
            
            try:
                # Prepare prompt
                content = item['content']
                prompt = prompt_template.replace("{content}", content)
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                
                # Call LLM
                response = await llm_client.call_llm(
                    messages=messages,
                    extra_params=extra_params
                )
                
                # Extract JSON data (handle possible Markdown format)
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response
                
                # Parse JSON
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list):
                        # Add metadata
                        for sample in data:
                            # Ensure required fields exist
                            if "instruction" not in sample:
                                sample["instruction"] = "Answer questions based on the content"
                            if "output" not in sample:
                                sample["output"] = "No answer"
                            if "input" not in sample:
                                sample["input"] = ""
                        results.extend(data)
                    else:
                        print(f"Returned data is not in list format, skipping")
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
                    # Try to fix JSON
                    fixed_data = llm_client.fix_json_response(json_str)
                    if fixed_data:
                        results.extend(fixed_data)
                    else:
                        print(f"Unable to fix JSON data, skipping")
                
            except Exception as e:
                print(f"Processing failed: {str(e)}")
                continue
    
    # Run async task
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(process_all())
    finally:
        loop.close()
    
    return results

def find_pdf_files(directory):
    """Traverse directory and find all PDF files"""
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def save_training_data(data, output_path):
    """Save training data to file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Training data saved to: {output_path}")

def main():
    """Main function"""
    # Fixed system prompt and prompt template
    system_prompt = "You are a professional data processing assistant responsible for converting document content into high-quality training data."
    prompt_template = """Please generate training data based on the following PDF content. Requirements:
1. Analyze the main content and structure of the document
2. Extract meaningful question-answer pairs or instruction-response pairs
3. Each sample should include clear instructions and high-quality answers
4. If the content is suitable for multi-turn dialogue, generate samples with history field
5. Ensure the data is valuable for LLM training
6. Only organize general information, not proprietary information (such as questions like "this article..." "this research..." etc.)
7. Ensure the generated training data content is in Chinese
8. Generate training data based on the complete content provided, do not miss any important information

Please strictly output in the following JSON format:
[
  {
    "instruction": "User instruction (required)",
    "input": "User input (optional)",
    "output": "Model answer (required)",
    "system": "System prompt (optional)",
    "history": [
      ["First round instruction (optional)", "First round answer (optional)"],
      ["Second round instruction (optional)", "Second round answer (optional)"]
    ]
  }
]
**Please note that each data item must contain at least the three fields "instruction", "input", and "output". Do not add any fields other than those mentioned above**
PDF content:
{content}"""
    
    # Get input parameters
    print("===== PDF Training Data Generator =====")
    pdf_dir = input("Please enter the PDF directory path: ").strip()
    
    # Verify if directory exists
    if not os.path.isdir(pdf_dir):
        print(f"Error: Directory '{pdf_dir}' does not exist")
        return
    
    # Find PDF files
    pdf_files = find_pdf_files(pdf_dir)
    if not pdf_files:
        print(f"Error: No PDF files found in directory '{pdf_dir}'")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Get model information - default to Alibaba Cloud DashScope
    print("===== Alibaba Cloud DashScope Configuration =====")
    provider = input("Please enter model name (e.g., qwen-plus): ").strip()
    if not provider:
        provider = "qwen-plus"  # Default to qwen-plus
        print(f"No input, using default model: {provider}")
    
    api_key = input("Please enter Alibaba Cloud DashScope API key: ").strip()
    if not api_key:
        print("Error: API key cannot be empty")
        return
    
    # Default to Alibaba Cloud DashScope compatible mode URL
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    print(f"Using default API base URL: {base_url}")
    print("=============================")
    
    # Get save path - default to current directory
    default_path = f"./training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = input(f"Please enter training data save path (default: {default_path}, press Enter to use default): ").strip()
    if not output_path:
        output_path = default_path
        print(f"Using default save path: {output_path}")
    
    # Process PDF files
    print("Starting to process PDF files...")
    contents = []
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"Processing PDF file {i}/{len(pdf_files)}: {pdf_path}")
        try:
            content = extract_text_from_pdf(pdf_path)
            contents.append(content)
        except Exception as e:
            print(f"Failed to process file {pdf_path}: {str(e)}")
    
    if not contents:
        print("Error: No PDF files were successfully processed")
        return
    
    # Generate training data
    print("Starting to generate training data...")
    try:
        training_data = generate_training_data(
            contents=contents,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            system_prompt=system_prompt,
            prompt_template=prompt_template
        )
        
        if training_data:
            # Save training data
            save_training_data(training_data, output_path)
            print(f"Successfully generated {len(training_data)} training data entries")
        else:
            print("Warning: No training data was generated")
    
    except Exception as e:
        print(f"Failed to generate training data: {str(e)}")

if __name__ == "__main__":
    main()