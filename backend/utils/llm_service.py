"""
LLM Service for intelligent data analysis and visualization code generation.
Uses local Llama 3.2 3B with 8-bit quantization.
Model is cached locally to avoid re-downloading.
"""

import os
import re
import json
import logging
import importlib.util
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
        self.load_error = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize Llama 3.2 3B from local disk with 4-bit quantization"""
        if not TORCH_AVAILABLE:
            self.load_error = "PyTorch/Transformers are not installed."
            logger.warning(f"{self.load_error} Using template fallback.")
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
                self.load_error = (
                    f"CUDA is not available to PyTorch. Installed torch build: {torch.__version__}. "
                    "Install a CUDA-enabled PyTorch build to use the local LLM."
                )
                logger.warning(self.load_error)
                return

            if importlib.util.find_spec("bitsandbytes") is None:
                self.load_error = "bitsandbytes is not installed; 8-bit local Llama loading is unavailable."
                logger.warning(self.load_error)
                return
            
            # Local model path
            local_model_path = os.path.join(MODEL_CACHE_DIR, "Llama-3.2-3B-Instruct")
            index_path = os.path.join(local_model_path, "model.safetensors.index.json")
            if not os.path.exists(index_path):
                self.load_error = f"Missing model index file: {index_path}"
                logger.warning(self.load_error)
                return

            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            expected_shards = sorted(set(index_data.get("weight_map", {}).values()))
            missing_shards = [
                shard for shard in expected_shards
                if not os.path.exists(os.path.join(local_model_path, shard))
            ]
            if missing_shards:
                self.load_error = (
                    "Local Llama weights are incomplete. Missing: "
                    + ", ".join(missing_shards[:5])
                )
                logger.warning(self.load_error)
                return
            
            # 8-bit quantization config with CPU offload for 4GB-class GPUs.
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            
            print(f"📂 Loading tokenizer from local disk: {local_model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            offload_folder = os.path.join(MODEL_CACHE_DIR, "offload")
            os.makedirs(offload_folder, exist_ok=True)
            device_map = {
                "model": 0,
                "lm_head": "cpu",
            }

            print("🚀 Loading model from local disk with 8-bit quantization on GPU/CPU offload...")
            self.model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                quantization_config=quantization_config,
                device_map=device_map,
                dtype="auto",
                offload_folder=offload_folder
            )
            
            self.model.eval()
            self.model_loaded = True
            self.load_error = None
            print("✅ Model ready. No downloads. GPU active. 8-bit with CPU offload enabled.")
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            self.load_error = str(e)
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

        template_result = self._template_generate_code(query, columns, numeric_cols, categorical_cols)
        if template_result and template_result.get('matched_intent'):
            template_result.pop('matched_intent', None)
            return template_result

        if self.model_loaded:
            result = self._llm_generate_code(query, columns, dtypes, numeric_cols, categorical_cols, schema_chunks)
            if result and result.get('code'):
                return result

        template_result.pop('matched_intent', None)
        return template_result

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
- Generate exactly ONE best visualization for the user's question
- Do not create subplots, dashboards, multiple figures, or extra charts
- Add title, xlabel, ylabel as appropriate
- Use good colors (steelblue, viridis palette, coolwarm colormap)

Return ONLY valid JSON in this exact format:
{{"code": "python code here", "explanation": "what the visualization shows", "visualization_type": "chart type"}}"""

        response = self._generate_with_llm(prompt, max_new_tokens=450)
        
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

        def find_numeric_column(terms):
            for term in terms:
                term_lower = term.lower()
                for col in numeric_cols:
                    col_lower = col.lower()
                    if col_lower in query_lower or (term_lower in query_lower and term_lower in col_lower):
                        return col
            return None

        def mentioned_numeric_columns():
            mentioned = []
            aliases = {
                'cgpa': ['cgpa', 'gpa', 'grade'],
                'score': ['score', 'aptitude', 'test', 'marks', 'exam']
            }
            for col in numeric_cols:
                col_lower = col.lower()
                if col_lower in query_lower:
                    mentioned.append(col)
                    continue
                for alias_key, terms in aliases.items():
                    if alias_key in col_lower and any(term in query_lower for term in terms):
                        mentioned.append(col)
                        break
            return list(dict.fromkeys(mentioned))

        relationship_intent = any(w in query_lower for w in [
            'scatter', 'vs', 'versus', 'against', 'relationship between',
            'relationship', 'correlate', 'correlation', 'tend to', 'tends to',
            'associated', 'higher', 'lower', 'increase', 'score higher',
            'score lower', 'aptitude'
        ])

        percentage_intent = any(w in query_lower for w in [
            'percentage', 'percent', 'proportion', 'ratio', 'share'
        ])

        def normalize_name(value):
            return re.sub(r'[^a-z0-9]+', '', str(value).lower())

        column_by_norm = {normalize_name(col): col for col in columns}

        alias_terms = {
            'medv': ['medv', 'price', 'prices', 'house price', 'house prices', 'housing price', 'housing prices'],
            'crim': ['crim', 'crime', 'crime rate'],
            'rm': ['rm', 'rooms', 'number of rooms', 'average number of rooms'],
            'nox': ['nox', 'nitric oxide'],
            'chas': ['chas', 'charles river'],
            'lstat': ['lstat', 'lower status'],
            'tax': ['tax', 'property tax'],
            'ptratio': ['ptratio', 'pupil-teacher', 'pupil teacher'],
            'rad': ['rad', 'accessibility'],
            'dis': ['dis', 'distance', 'employment centers'],
            'age': ['age', 'old', 'newer'],
            'indus': ['indus', 'industrial land'],
            'zn': ['zn', 'zoning'],
        }

        def find_column_by_alias(alias_key):
            if normalize_name(alias_key) in column_by_norm:
                return column_by_norm[normalize_name(alias_key)]
            for col in columns:
                if normalize_name(alias_key) == normalize_name(col):
                    return col
            return None

        def mentioned_columns_from_query():
            found = []
            for raw in re.findall(r'\(([A-Za-z0-9_ .-]+)\)', query):
                column = column_by_norm.get(normalize_name(raw))
                if column and column not in found:
                    found.append(column)
            for col in columns:
                col_norm = normalize_name(col)
                if col_norm and re.search(rf'\b{re.escape(str(col).lower())}\b', query_lower) and col not in found:
                    found.append(col)
            for alias_key, terms in alias_terms.items():
                column = find_column_by_alias(alias_key)
                if not column or column in found:
                    continue
                if any(term in query_lower for term in terms):
                    found.append(column)
            return found

        mentioned_cols = mentioned_columns_from_query()

        def first_numeric(cols):
            for col in cols:
                if col in numeric_cols:
                    return col
            return numeric_cols[0] if numeric_cols else None

        def target_value_column(exclude=None):
            exclude = set(exclude or [])
            price_col = find_column_by_alias('medv')
            if price_col and price_col not in exclude:
                return price_col
            for col in mentioned_cols:
                if col in numeric_cols and col not in exclude:
                    return col
            for col in numeric_cols:
                if col not in exclude:
                    return col
            return None

        def dimension_column(exclude=None):
            exclude = set(exclude or [])
            for col in mentioned_cols:
                if col not in exclude:
                    return col
            for col in categorical_cols + numeric_cols:
                if col not in exclude:
                    return col
            return None

        def xy_columns():
            price_col = find_column_by_alias('medv')
            if price_col in mentioned_cols and len(mentioned_cols) > 1:
                x_col = next((col for col in mentioned_cols if col != price_col and col in numeric_cols), None)
                if x_col:
                    return x_col, price_col
            x_col = next((col for col in mentioned_cols if col in numeric_cols), None)
            y_col = target_value_column(exclude=[x_col])
            return x_col, y_col

        def label_expr(col):
            if normalize_name(col) in ['placed', 'chas']:
                positive = 'Placed' if normalize_name(col) == 'placed' else 'Near Charles River'
                negative = 'Not Placed' if normalize_name(col) == 'placed' else 'Not Near Charles River'
                return f"df['{col}'].map({{1: '{positive}', 0: '{negative}', '1': '{positive}', '0': '{negative}'}}).fillna(df['{col}'].astype(str))"
            return f"df['{col}'].astype(str)"

        plot_type = None
        if 'pie' in query_lower:
            plot_type = 'pie'
        elif 'horizontal bar' in query_lower:
            plot_type = 'horizontal_bar'
        elif 'violin' in query_lower:
            plot_type = 'violin'
        elif 'box' in query_lower or 'outlier' in query_lower:
            plot_type = 'box'
        elif 'density' in query_lower or 'kde' in query_lower:
            plot_type = 'density'
        elif 'line' in query_lower or 'change with' in query_lower:
            plot_type = 'line'
        elif 'heatmap' in query_lower:
            plot_type = 'heatmap'
        elif 'regression' in query_lower or 'regplot' in query_lower:
            plot_type = 'regression'
        elif 'scatter' in query_lower or 'relationship' in query_lower or 'vary with' in query_lower:
            plot_type = 'scatter'
        elif 'bar' in query_lower or 'compare' in query_lower:
            plot_type = 'bar'
        elif 'histogram' in query_lower or 'distribution' in query_lower or 'frequency' in query_lower:
            plot_type = 'histogram'

        if plot_type == 'histogram':
            target = first_numeric(mentioned_cols)
            if target:
                code = f"""
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='{target}', kde=True, color='steelblue', edgecolor='white')
plt.title('Distribution of {target}', fontsize=14, fontweight='bold')
plt.xlabel('{target}')
plt.ylabel('Frequency')
plt.tight_layout()
"""
                return {'code': code, 'explanation': f'Histogram showing the distribution of {target}.', 'visualization_type': 'histogram', 'matched_intent': True}

        if plot_type == 'density':
            target = first_numeric(mentioned_cols)
            if target:
                code = f"""
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='{target}', fill=True, color='steelblue', linewidth=2)
plt.title('Density Distribution of {target}', fontsize=14, fontweight='bold')
plt.xlabel('{target}')
plt.ylabel('Density')
plt.tight_layout()
"""
                return {'code': code, 'explanation': f'Density plot showing the distribution of {target}.', 'visualization_type': 'density', 'matched_intent': True}

        if plot_type in ['scatter', 'regression']:
            x, y = xy_columns()
            if x in numeric_cols and y in numeric_cols:
                plot_call = "sns.regplot(data=plot_df, x='" + x + "', y='" + y + "', scatter_kws={'alpha': 0.45, 'color': 'steelblue'}, line_kws={'color': 'red', 'linestyle': '--', 'linewidth': 2})" if plot_type == 'regression' else "sns.scatterplot(data=plot_df, x='" + x + "', y='" + y + "', alpha=0.45, color='steelblue', edgecolor=None)"
                code = f"""
plt.figure(figsize=(10, 6))
plot_df = df[['{x}', '{y}']].dropna()
{plot_call}
plt.title('{x} vs {y}', fontsize=14, fontweight='bold')
plt.xlabel('{x}')
plt.ylabel('{y}')
plt.tight_layout()
"""
                return {'code': code, 'explanation': f'{plot_type.title()} plot showing {x} versus {y}.', 'visualization_type': plot_type, 'matched_intent': True}

        if plot_type in ['bar', 'horizontal_bar']:
            y = target_value_column()
            x = dimension_column(exclude=[y])
            agg = 'median' if 'median' in query_lower else 'mean'
            if x and y and y in numeric_cols:
                orient = "h" if plot_type == 'horizontal_bar' else "v"
                if x in numeric_cols:
                    category_line = f"plot_df['{x}_range'] = pd.cut(plot_df['{x}'], bins=5)"
                    group_col = f"{x}_range"
                else:
                    category_line = f"plot_df['{x}_group'] = {label_expr(x)}"
                    group_col = f"{x}_group"
                if orient == "h":
                    plot_line = f"sns.barplot(data=summary, x='{y}', y='{group_col}', palette='viridis')"
                    axis_labels = f"plt.xlabel('{agg.title()} {y}')\nplt.ylabel('{x}')"
                else:
                    plot_line = f"sns.barplot(data=summary, x='{group_col}', y='{y}', palette='viridis')"
                    axis_labels = f"plt.xlabel('{x}')\nplt.ylabel('{agg.title()} {y}')\nplt.xticks(rotation=35, ha='right')"
                code = f"""
plt.figure(figsize=(10, 6))
plot_df = df[['{x}', '{y}']].dropna().copy()
{category_line}
summary = plot_df.groupby('{group_col}', observed=False)['{y}'].{agg}().reset_index()
summary['{group_col}'] = summary['{group_col}'].astype(str)
{plot_line}
plt.title('{agg.title()} {y} by {x}', fontsize=14, fontweight='bold')
{axis_labels}
plt.tight_layout()
"""
                return {'code': code, 'explanation': f'{plot_type.replace("_", " ").title()} chart comparing {agg} {y} across {x}.', 'visualization_type': plot_type, 'matched_intent': True}

        if plot_type == 'pie':
            target = dimension_column()
            if target:
                code = f"""
plt.figure(figsize=(8, 8))
labels = {label_expr(target)}
counts = labels.value_counts().head(8)
colors = sns.color_palette('viridis', len(counts))
plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Percentage Distribution of {target}', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
"""
                return {'code': code, 'explanation': f'Pie chart showing percentage distribution of {target}.', 'visualization_type': 'pie', 'matched_intent': True}

        if plot_type in ['box', 'violin']:
            y = target_value_column()
            x = dimension_column(exclude=[y])
            if y:
                if x and x != y:
                    if x in numeric_cols:
                        category_line = f"plot_df['{x}_group'] = pd.cut(plot_df['{x}'], bins=2, labels=['Lower {x}', 'Higher {x}'])"
                    else:
                        category_line = f"plot_df['{x}_group'] = {label_expr(x)}"
                    plot_func = 'sns.violinplot' if plot_type == 'violin' else 'sns.boxplot'
                    code = f"""
plt.figure(figsize=(10, 6))
plot_df = df[['{x}', '{y}']].dropna().copy()
{category_line}
{plot_func}(data=plot_df, x='{x}_group', y='{y}', palette='viridis')
plt.title('{y} by {x}', fontsize=14, fontweight='bold')
plt.xlabel('{x}')
plt.ylabel('{y}')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
"""
                else:
                    plot_func = 'sns.violinplot' if plot_type == 'violin' else 'sns.boxplot'
                    code = f"""
plt.figure(figsize=(8, 6))
{plot_func}(data=df, y='{y}', color='steelblue')
plt.title('{plot_type.title()} Plot of {y}', fontsize=14, fontweight='bold')
plt.ylabel('{y}')
plt.tight_layout()
"""
                return {'code': code, 'explanation': f'{plot_type.title()} plot for {y}.', 'visualization_type': plot_type, 'matched_intent': True}

        if plot_type == 'line':
            x, y = xy_columns()
            if x in numeric_cols and y in numeric_cols:
                code = f"""
plt.figure(figsize=(10, 6))
plot_df = df[['{x}', '{y}']].dropna().sort_values('{x}')
plot_df['{x}_bin'] = pd.cut(plot_df['{x}'], bins=10)
summary = plot_df.groupby('{x}_bin', observed=False).agg({{'{x}': 'mean', '{y}': 'mean'}}).dropna()
sns.lineplot(data=summary, x='{x}', y='{y}', marker='o', color='steelblue')
plt.title('Average {y} by Increasing {x}', fontsize=14, fontweight='bold')
plt.xlabel('{x}')
plt.ylabel('Average {y}')
plt.tight_layout()
"""
                return {'code': code, 'explanation': f'Line chart showing how average {y} changes with {x}.', 'visualization_type': 'line', 'matched_intent': True}

        if plot_type == 'heatmap':
            heat_cols = [col for col in mentioned_cols if col in numeric_cols]
            if len(heat_cols) < 2:
                heat_cols = numeric_cols[:2]
            if len(heat_cols) >= 2:
                code = f"""
plt.figure(figsize=(8, 6))
corr = df[{heat_cols}].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
"""
                return {'code': code, 'explanation': f'Heatmap showing correlation between {", ".join(heat_cols)}.', 'visualization_type': 'heatmap', 'matched_intent': True}

        placed_col = next((col for col in columns if col.lower() == 'placed'), None)
        if placed_col and percentage_intent and 'placed' in query_lower:
            if 'pie' in query_lower:
                code = f"""
plt.figure(figsize=(8, 8))
status = df['{placed_col}'].map({{1: 'Placed', 0: 'Not Placed', '1': 'Placed', '0': 'Not Placed'}}).fillna(df['{placed_col}'].astype(str))
counts = status.value_counts()
colors = sns.color_palette('viridis', len(counts))
plt.pie(
    counts.values,
    labels=counts.index,
    autopct=lambda pct: f'{{pct:.1f}}%\\n(n={{int(round(pct / 100 * counts.sum()))}})',
    startangle=90,
    colors=colors,
    textprops={{'fontsize': 11, 'fontweight': 'bold'}}
)
plt.title('Placed vs Not Placed Students', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
"""
                return {
                    'code': code,
                    'explanation': f'Pie chart showing the percentage of students by {placed_col} status.',
                    'visualization_type': 'pie',
                    'matched_intent': True
                }

            code = f"""
plt.figure(figsize=(8, 6))
status = df['{placed_col}'].map({{1: 'Placed', 0: 'Not Placed', '1': 'Placed', '0': 'Not Placed'}}).fillna(df['{placed_col}'].astype(str))
percentages = status.value_counts(normalize=True).mul(100)
counts = status.value_counts()
colors = sns.color_palette('viridis', len(percentages))
ax = sns.barplot(x=percentages.index, y=percentages.values, palette=colors)
plt.title('Placement Percentage', fontsize=14, fontweight='bold')
plt.xlabel('Placement Status')
plt.ylabel('Percentage of Students')
plt.ylim(0, max(100, percentages.max() + 10))
for i, (label, pct) in enumerate(percentages.items()):
    ax.text(i, pct + 1, f'{{pct:.1f}}%\\n(n={{counts[label]}})', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
"""
            return {
                'code': code,
                'explanation': f'Bar chart showing the percentage of students by {placed_col} status.',
                'visualization_type': 'bar',
                'matched_intent': True
            }
        
        if any(w in query_lower for w in ['heatmap', 'correlation matrix']):
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
                return {'code': code, 'explanation': 'Correlation heatmap showing relationships between numeric variables', 'visualization_type': 'heatmap', 'matched_intent': True}
        
        elif relationship_intent:
            if len(numeric_cols) >= 2:
                mentioned = mentioned_numeric_columns()
                x = find_numeric_column(['cgpa', 'gpa', 'grade']) or (mentioned[0] if mentioned else numeric_cols[0])
                y = find_numeric_column(['score', 'aptitude', 'test', 'marks', 'exam']) or (
                    mentioned[1] if len(mentioned) > 1 else next((col for col in numeric_cols if col != x), numeric_cols[0])
                )
                if x == y:
                    y = next((col for col in numeric_cols if col != x), y)
                code = f"""
plt.figure(figsize=(10, 6))
plot_df = df[['{x}', '{y}']].dropna()
corr = plot_df['{x}'].corr(plot_df['{y}'])
sns.scatterplot(data=plot_df, x='{x}', y='{y}', alpha=0.45, color='steelblue', edgecolor=None)
sns.regplot(data=plot_df, x='{x}', y='{y}', scatter=False, color='red', line_kws={{'linestyle': '--', 'linewidth': 2}})
plt.title('{x} vs {y} (Pearson r = ' + f'{{corr:.3f}}' + ')', fontsize=14, fontweight='bold')
plt.xlabel('{x}')
plt.ylabel('{y}')
plt.text(0.02, 0.98, 'Strong positive relationship' if corr >= 0.7 else 'Moderate positive relationship' if corr >= 0.3 else 'Weak/no positive relationship',
         transform=plt.gca().transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
plt.tight_layout()
"""
                return {
                    'code': code,
                    'explanation': f'Scatter plot with regression line and Pearson correlation for {x} and {y}.',
                    'visualization_type': 'scatter',
                    'matched_intent': True
                }

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
                return {'code': code, 'explanation': f'Distribution histogram with KDE, mean and median lines for {target}', 'visualization_type': 'histogram', 'matched_intent': True}
        
        elif any(w in query_lower for w in ['scatter', 'plot']):
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
                return {'code': code, 'explanation': f'Scatter plot with regression line showing relationship between {x} and {y}', 'visualization_type': 'scatter', 'matched_intent': True}
        
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
                return {'code': code, 'explanation': f'Bar chart showing top categories in {target}', 'visualization_type': 'bar', 'matched_intent': True}
        
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
            return {'code': code, 'explanation': 'Box plot showing distribution and outliers', 'visualization_type': 'boxplot', 'matched_intent': True}
        
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
                return {'code': code, 'explanation': f'Pie chart showing proportions of {target}', 'visualization_type': 'pie', 'matched_intent': True}
        
        # Default: create one simple overview plot instead of a multi-chart dashboard.
        if numeric_cols:
            col1 = numeric_cols[0]
            code = f"""
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='{col1}', kde=True, color='steelblue', edgecolor='white')
plt.title('Distribution of {col1}', fontsize=14, fontweight='bold')
plt.xlabel('{col1}')
plt.ylabel('Count')
plt.tight_layout()
"""
            return {'code': code, 'explanation': f'Distribution of {col1}', 'visualization_type': 'histogram', 'matched_intent': False}
        
        return {'code': 'plt.figure()\nplt.text(0.5, 0.5, "No suitable data to visualize", ha="center", va="center", fontsize=14)\nplt.axis("off")', 
                'explanation': 'No suitable numeric data found for visualization', 'visualization_type': 'empty', 'matched_intent': False}

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
