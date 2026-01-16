"""
LLM Service for intelligent data analysis and visualization code generation.
Uses local Llama 3.1 3B with 4-bit quantization.
Model is cached locally to avoid re-downloading.
"""

import os
import re
import json
import logging
from typing import Dict, List, Union, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Model cache directory - saves model locally
MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# Try to import torch and transformers
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch/Transformers not available. Install with: pip install torch transformers bitsandbytes")


class LLMService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.device = None
        self.dataset_context = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize Llama 3.2 3B from local disk with 4-bit quantization"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Using template fallback.")
            return
            
        try:
            # Check CUDA
            if torch.cuda.is_available():
                self.device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"CUDA available: {gpu_name} ({vram:.1f}GB)")
                print(f"🖥️  GPU: {gpu_name} ({vram:.1f}GB VRAM)")
            else:
                logger.warning("CUDA not available. Local LLM requires GPU.")
                return
            
            # Local model path
            local_model_path = os.path.join(MODEL_CACHE_DIR, "Llama-3.2-3B-Instruct")
            
            # 4-bit quantization config (Step 2: Daily-use)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            print(f"📂 Loading tokenizer from local disk: {local_model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print("🚀 Loading model from local disk with 4-bit quantization on GPU...")
            self.model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                quantization_config=quantization_config,
                device_map="auto"
            )
            
            self.model.eval()
            self.model_loaded = True
            print("✅ Model ready. No downloads. GPU active. 4-bit enabled.")
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            print(f"❌ Failed to load model: {e}")
            print("Using template-based fallback.")

    def _generate_with_llm(self, prompt: str, max_new_tokens: int = 500) -> Optional[str]:
        """Generate text using the local Llama model"""
        if not self.model_loaded:
            return None
            
        try:
            # Format for Llama 3.1 Instruct
            messages = [
                {"role": "system", "content": "You are a data visualization expert. Generate Python code using pandas, matplotlib, and seaborn. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template
            input_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only new tokens
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return None

    def analyze_dataset(self, df: pd.DataFrame) -> Dict:
        """Analyze dataset structure"""
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
            'numeric_cols': list(df.select_dtypes(include=['int64', 'float64']).columns),
            'categorical_cols': list(df.select_dtypes(include=['object', 'category']).columns),
            'missing_values': df.isnull().sum().to_dict(),
        }
        self.dataset_context = info
        
        # Try LLM summary
        if self.model_loaded:
            prompt = f"""Analyze this dataset briefly (2 sentences max):
Columns: {', '.join(info['columns'][:10])}
Shape: {info['shape'][0]} rows, {info['shape'][1]} columns
Numeric: {', '.join(info['numeric_cols'][:5])}
Categorical: {', '.join(info['categorical_cols'][:5])}"""
            
            summary = self._generate_with_llm(prompt, max_new_tokens=100)
            if summary:
                info['llm_summary'] = summary
        
        return info

    def generate_visualization_code(self, query: str, df: pd.DataFrame, schema_chunks: list = None) -> Dict:
        """Generate Python visualization code based on user query."""
        columns = list(df.columns)
        dtypes = {col: str(df[col].dtype) for col in df.columns}
        numeric_cols = list(df.select_dtypes(include=['int64', 'float64']).columns)
        categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
        
        if self.model_loaded:
            result = self._llm_generate_code(query, columns, dtypes, numeric_cols, categorical_cols, schema_chunks)
            if result and result.get('code'):
                return result
        
        return self._template_generate_code(query, columns, numeric_cols, categorical_cols)

    def _llm_generate_code(self, query: str, columns: list, dtypes: dict, 
                          numeric_cols: list, categorical_cols: list, schema_chunks: list = None) -> Optional[Dict]:
        """Use local Llama to generate visualization code."""
        
        schema_context = "\n".join(schema_chunks) if schema_chunks else "Standard pandas dataframe."
        
        prompt = f"""Handle NaNs (median/ffill). Schema: {schema_context}. Query: {query}
        
The DataFrame 'df' has these columns: {', '.join(columns[:15])}
Numeric columns: {', '.join(numeric_cols[:8]) if numeric_cols else 'None'}
Categorical columns: {', '.join(categorical_cols[:5]) if categorical_cols else 'None'}

Requirements:
- Libraries already imported: plt (matplotlib.pyplot), sns (seaborn), pd (pandas), np (numpy)
- DataFrame variable is 'df'
- Use plt.figure(figsize=(10,6)) for single plots
- Add title, xlabel, ylabel as appropriate
- Use good colors (steelblue, viridis palette, coolwarm colormap)

Return ONLY valid JSON in this exact format:
{{"code": "python code here", "explanation": "what the visualization shows", "visualization_type": "chart type"}}"""

        response = self._generate_with_llm(prompt, max_new_tokens=800)
        
        if response:
            try:
                # Try to extract JSON from response
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    result = json.loads(json_match.group())
                    if 'code' in result:
                        # Clean up escape sequences
                        result['code'] = result['code'].replace('\\n', '\n').replace('\\t', '    ')
                        logger.info(f"LLM generated: {result.get('visualization_type')}")
                        return result
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM response as JSON: {e}")
        
        return None

    def _template_generate_code(self, query: str, columns: list, 
                                numeric_cols: list, categorical_cols: list) -> Dict:
        """Template-based code generation fallback."""
        query_lower = query.lower()
        
        if any(w in query_lower for w in ['correlation', 'heatmap', 'relationship', 'correlate']):
            if len(numeric_cols) >= 2:
                cols_to_use = numeric_cols[:10]  # Limit for readability
                code = f"""
plt.figure(figsize=(12, 10))
correlation_matrix = df[{cols_to_use}].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
"""
                return {'code': code, 'explanation': 'Correlation heatmap showing relationships between numeric variables', 'visualization_type': 'heatmap'}
        
        elif any(w in query_lower for w in ['distribution', 'histogram', 'dist', 'spread']):
            target = None
            for col in numeric_cols:
                if col.lower() in query_lower:
                    target = col
                    break
            if not target and numeric_cols:
                target = numeric_cols[0]
            if target:
                code = f"""
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='{target}', kde=True, color='steelblue', edgecolor='white')
plt.axvline(df['{target}'].mean(), color='red', linestyle='--', label=f'Mean: {{df["{target}"].mean():.2f}}')
plt.axvline(df['{target}'].median(), color='green', linestyle='--', label=f'Median: {{df["{target}"].median():.2f}}')
plt.title('Distribution of {target}', fontsize=14, fontweight='bold')
plt.xlabel('{target}')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
"""
                return {'code': code, 'explanation': f'Distribution histogram with KDE, mean and median lines for {target}', 'visualization_type': 'histogram'}
        
        elif any(w in query_lower for w in ['scatter', 'plot', 'vs', 'versus', 'against', 'relationship between']):
            if len(numeric_cols) >= 2:
                x, y = numeric_cols[0], numeric_cols[1]
                for col in numeric_cols:
                    if col.lower() in query_lower:
                        if x == numeric_cols[0]:
                            x = col
                        else:
                            y = col
                code = f"""
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='{x}', y='{y}', alpha=0.6, color='steelblue')
sns.regplot(data=df, x='{x}', y='{y}', scatter=False, color='red', line_kws={{'linestyle': '--', 'linewidth': 2}})
plt.title('{x} vs {y}', fontsize=14, fontweight='bold')
plt.xlabel('{x}')
plt.ylabel('{y}')
plt.tight_layout()
"""
                return {'code': code, 'explanation': f'Scatter plot with regression line showing relationship between {x} and {y}', 'visualization_type': 'scatter'}
        
        elif any(w in query_lower for w in ['bar', 'category', 'count', 'categorical', 'top']):
            target = None
            for col in categorical_cols:
                if col.lower() in query_lower:
                    target = col
                    break
            if not target and categorical_cols:
                target = categorical_cols[0]
            if target:
                code = f"""
plt.figure(figsize=(10, 6))
counts = df['{target}'].value_counts().head(10)
colors = sns.color_palette('viridis', len(counts))
sns.barplot(x=counts.values, y=counts.index, palette=colors)
plt.title('Top Categories in {target}', fontsize=14, fontweight='bold')
plt.xlabel('Count')
plt.ylabel('{target}')
for i, v in enumerate(counts.values):
    plt.text(v + 0.5, i, str(v), va='center')
plt.tight_layout()
"""
                return {'code': code, 'explanation': f'Bar chart showing top categories in {target}', 'visualization_type': 'bar'}
        
        elif any(w in query_lower for w in ['box', 'boxplot', 'outlier', 'quartile']):
            target = None
            for col in numeric_cols:
                if col.lower() in query_lower:
                    target = col
                    break
            
            if target:
                code = f"""
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, y='{target}', color='steelblue')
plt.title('Box Plot of {target}', fontsize=14, fontweight='bold')
plt.ylabel('{target}')
plt.tight_layout()
"""
            else:
                cols = numeric_cols[:6]
                code = f"""
plt.figure(figsize=(14, 6))
df[{cols}].boxplot(patch_artist=True)
plt.title('Box Plots of Numeric Features', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Value')
plt.tight_layout()
"""
            return {'code': code, 'explanation': 'Box plot showing distribution and outliers', 'visualization_type': 'boxplot'}
        
        elif any(w in query_lower for w in ['pie', 'proportion', 'percentage']):
            target = None
            for col in categorical_cols:
                if col.lower() in query_lower:
                    target = col
                    break
            if not target and categorical_cols:
                target = categorical_cols[0]
            if target:
                code = f"""
plt.figure(figsize=(10, 8))
counts = df['{target}'].value_counts().head(8)
colors = sns.color_palette('viridis', len(counts))
plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Distribution of {target}', fontsize=14, fontweight='bold')
plt.tight_layout()
"""
                return {'code': code, 'explanation': f'Pie chart showing proportions of {target}', 'visualization_type': 'pie'}
        
        # Default: create a comprehensive overview
        if numeric_cols:
            col1 = numeric_cols[0]
            col2 = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
            code = f"""
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribution
sns.histplot(data=df, x='{col1}', kde=True, ax=axes[0, 0], color='steelblue')
axes[0, 0].set_title('Distribution of {col1}')

# Box plots
df[{numeric_cols[:5]}].boxplot(ax=axes[0, 1])
axes[0, 1].set_title('Box Plots')
axes[0, 1].tick_params(axis='x', rotation=45)

# Correlation
corr = df[{numeric_cols[:8]}].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[1, 0], fmt='.2f', center=0)
axes[1, 0].set_title('Correlation Matrix')

# Scatter
sns.scatterplot(data=df, x='{col1}', y='{col2}', ax=axes[1, 1], alpha=0.6)
axes[1, 1].set_title('{col1} vs {col2}')

plt.tight_layout()
"""
            return {'code': code, 'explanation': 'Comprehensive data overview with distribution, box plots, correlation, and scatter plot', 'visualization_type': 'dashboard'}
        
        return {'code': 'plt.figure()\nplt.text(0.5, 0.5, "No suitable data to visualize", ha="center", va="center", fontsize=14)\nplt.axis("off")', 
                'explanation': 'No suitable numeric data found for visualization', 'visualization_type': 'empty'}

    def understand_query(self, query: str, available_columns: List[str]) -> Dict:
        """Legacy query understanding for filtering."""
        query_lower = query.lower()
        mentioned = [c for c in available_columns if c.lower() in query_lower]
        
        viz_types = []
        if any(w in query_lower for w in ['correlation', 'heatmap']):
            viz_types.append('correlation_heatmap')
        if any(w in query_lower for w in ['distribution', 'histogram']):
            viz_types.append('dist_')
        if any(w in query_lower for w in ['box']):
            viz_types.append('box_plots')
        if any(w in query_lower for w in ['scatter']):
            viz_types.append('scatter_matrix')
        if any(w in query_lower for w in ['bar']):
            viz_types.append('bar_')
        
        return {
            'columns': mentioned if mentioned else available_columns,
            'visualization_types': viz_types if viz_types else 'all'
        }


llm_service = None

def get_llm_service():
    global llm_service
    if llm_service is None:
        llm_service = LLMService()
    return llm_service
