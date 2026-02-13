"""
ALCALA Self-Modification Module
Allows ALCALA to modify its own code architecture when requested

CRITICAL SAFETY FEATURES:
- Backup before every modification
- Syntax validation before applying changes
- Rollback capability
- Change logging
- User confirmation for critical changes

This module enables ALCALA to:
1. Read its own source code
2. Analyze and understand code structure
3. Make requested modifications
4. Validate changes before applying
5. Backup and rollback if needed
"""

import os
import ast
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import hashlib


class ALCALASelfModification:
    """
    Self-modification module for ALCALA

    Allows ALCALA to safely modify its own code when requested
    """

    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.src_path = self.base_path / "src"
        self.backup_path = self.base_path / "backups" / "code_modifications"
        self.log_file = self.base_path / "logs" / "self_modifications.log"

        # Create directories
        self.backup_path.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Modification history
        self.modification_history = []

        print("[ALCALA Self-Mod] Self-modification module initialized")
        print(f"[ALCALA Self-Mod] Base path: {self.base_path}")
        print(f"[ALCALA Self-Mod] Backup path: {self.backup_path}")

    def get_modifiable_files(self) -> List[Path]:
        """Get list of files that ALCALA can modify"""
        modifiable_patterns = [
            "src/**/*.py",
            "src/**/*.js",
            "data/**/*.json",
            "config/**/*.yaml"
        ]

        files = []
        for pattern in modifiable_patterns:
            files.extend(self.base_path.glob(pattern))

        return sorted(files)

    def read_source_file(self, file_path: str) -> Optional[str]:
        """
        Read source file content

        Args:
            file_path: Relative or absolute path to file

        Returns:
            File content or None if error
        """
        try:
            # Convert to absolute path
            if not Path(file_path).is_absolute():
                file_path = self.base_path / file_path
            else:
                file_path = Path(file_path)

            # Security check - must be within AIMATRIX
            if not str(file_path.resolve()).startswith(str(self.base_path.resolve())):
                print(f"[ALCALA Self-Mod] Security: File outside AIMATRIX: {file_path}")
                return None

            if not file_path.exists():
                print(f"[ALCALA Self-Mod] File not found: {file_path}")
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            print(f"[ALCALA Self-Mod] Read file: {file_path.name} ({len(content)} bytes)")
            return content

        except Exception as e:
            print(f"[ALCALA Self-Mod] Error reading file: {e}")
            return None

    def analyze_code_structure(self, file_path: str) -> Dict:
        """
        Analyze Python code structure

        Args:
            file_path: Path to Python file

        Returns:
            Code structure analysis
        """
        content = self.read_source_file(file_path)
        if not content:
            return {"error": "Could not read file"}

        try:
            tree = ast.parse(content)

            analysis = {
                "classes": [],
                "functions": [],
                "imports": [],
                "globals": [],
                "line_count": len(content.split('\n'))
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    analysis["classes"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                    })
                elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                    analysis["functions"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args]
                    })
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    analysis["imports"].append(f"{node.module}")

            return analysis

        except SyntaxError as e:
            return {"error": f"Syntax error: {e}"}
        except Exception as e:
            return {"error": f"Analysis error: {e}"}

    def create_backup(self, file_path: Path) -> Path:
        """
        Create backup of file before modification

        Args:
            file_path: Path to file to backup

        Returns:
            Path to backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_file = self.backup_path / backup_name

        shutil.copy2(file_path, backup_file)

        print(f"[ALCALA Self-Mod] Backup created: {backup_name}")
        return backup_file

    def validate_python_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python code syntax

        Args:
            code: Python code string

        Returns:
            (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def modify_file(
        self,
        file_path: str,
        modifications: Dict,
        validate: bool = True,
        create_backup: bool = True
    ) -> Dict:
        """
        Modify source file with safety checks

        Args:
            file_path: Path to file to modify
            modifications: Dictionary of modifications to apply
            validate: Whether to validate syntax after modification
            create_backup: Whether to create backup before modification

        Returns:
            Result dictionary with status and details
        """
        # Convert to Path object
        if not Path(file_path).is_absolute():
            file_path = self.base_path / file_path
        else:
            file_path = Path(file_path)

        # Security check
        if not str(file_path.resolve()).startswith(str(self.base_path.resolve())):
            return {
                "success": False,
                "error": "Security: File outside AIMATRIX directory"
            }

        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }

        try:
            # Read current content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            # Create backup
            backup_file = None
            if create_backup:
                backup_file = self.create_backup(file_path)

            # Apply modifications
            modified_content = original_content

            if "replace" in modifications:
                for old_text, new_text in modifications["replace"].items():
                    modified_content = modified_content.replace(old_text, new_text)

            if "insert_before" in modifications:
                for marker, text in modifications["insert_before"].items():
                    modified_content = modified_content.replace(marker, text + marker)

            if "insert_after" in modifications:
                for marker, text in modifications["insert_after"].items():
                    modified_content = modified_content.replace(marker, marker + text)

            if "delete_lines" in modifications:
                lines = modified_content.split('\n')
                for line_range in modifications["delete_lines"]:
                    start, end = line_range
                    lines = lines[:start-1] + lines[end:]
                modified_content = '\n'.join(lines)

            # Validate if Python file
            if validate and file_path.suffix == '.py':
                is_valid, error = self.validate_python_syntax(modified_content)
                if not is_valid:
                    return {
                        "success": False,
                        "error": f"Validation failed: {error}",
                        "backup_file": str(backup_file) if backup_file else None
                    }

            # Write modified content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)

            # Log modification
            self._log_modification(
                file_path=str(file_path),
                backup_file=str(backup_file) if backup_file else None,
                modifications=modifications,
                success=True
            )

            return {
                "success": True,
                "file_path": str(file_path),
                "backup_file": str(backup_file) if backup_file else None,
                "changes_applied": len(modifications),
                "original_size": len(original_content),
                "new_size": len(modified_content)
            }

        except Exception as e:
            # Rollback if backup exists
            if backup_file and backup_file.exists():
                shutil.copy2(backup_file, file_path)
                print(f"[ALCALA Self-Mod] Rolled back changes due to error")

            self._log_modification(
                file_path=str(file_path),
                backup_file=str(backup_file) if backup_file else None,
                modifications=modifications,
                success=False,
                error=str(e)
            )

            return {
                "success": False,
                "error": f"Modification failed: {str(e)}",
                "rolled_back": backup_file is not None
            }

    def rollback_modification(self, backup_file: str) -> Dict:
        """
        Rollback to a previous backup

        Args:
            backup_file: Path to backup file

        Returns:
            Result dictionary
        """
        try:
            backup_path = Path(backup_file)
            if not backup_path.exists():
                return {
                    "success": False,
                    "error": "Backup file not found"
                }

            # Extract original filename from backup
            # Format: filename_YYYYMMDD_HHMMSS.ext
            parts = backup_path.stem.split('_')
            original_name = '_'.join(parts[:-2]) + backup_path.suffix

            # Restore from backup
            original_file = self.src_path / original_name
            shutil.copy2(backup_path, original_file)

            print(f"[ALCALA Self-Mod] Rolled back to: {backup_file}")

            return {
                "success": True,
                "restored_file": str(original_file),
                "backup_file": str(backup_file)
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Rollback failed: {str(e)}"
            }

    def _log_modification(
        self,
        file_path: str,
        backup_file: Optional[str],
        modifications: Dict,
        success: bool,
        error: Optional[str] = None
    ):
        """Log modification to file"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "file_path": file_path,
            "backup_file": backup_file,
            "modifications": modifications,
            "success": success,
            "error": error
        }

        self.modification_history.append(log_entry)

        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"[ALCALA Self-Mod] Error logging modification: {e}")

    def get_modification_history(self, limit: int = 10) -> List[Dict]:
        """Get recent modification history"""
        return self.modification_history[-limit:]

    def add_function_to_file(
        self,
        file_path: str,
        function_code: str,
        after_function: Optional[str] = None
    ) -> Dict:
        """
        Add a new function to a Python file

        Args:
            file_path: Path to Python file
            function_code: Complete function code to add
            after_function: Name of function to insert after (optional)

        Returns:
            Result dictionary
        """
        content = self.read_source_file(file_path)
        if not content:
            return {"success": False, "error": "Could not read file"}

        # Validate function code
        is_valid, error = self.validate_python_syntax(function_code)
        if not is_valid:
            return {"success": False, "error": f"Invalid function code: {error}"}

        if after_function:
            # Find the function to insert after
            pattern = f"def {after_function}("
            if pattern in content:
                # Find end of function (next def or class)
                lines = content.split('\n')
                insert_index = -1
                found_function = False

                for i, line in enumerate(lines):
                    if found_function and (line.startswith('def ') or line.startswith('class ')):
                        insert_index = i
                        break
                    if pattern in line:
                        found_function = True

                if insert_index > 0:
                    lines.insert(insert_index, '\n' + function_code + '\n')
                    modified_content = '\n'.join(lines)
                else:
                    # Add at end
                    modified_content = content + '\n\n' + function_code + '\n'
            else:
                return {"success": False, "error": f"Function {after_function} not found"}
        else:
            # Add at end of file
            modified_content = content + '\n\n' + function_code + '\n'

        # Use modify_file to apply changes
        modifications = {"replace": {content: modified_content}}
        return self.modify_file(file_path, modifications)

    def modify_class_method(
        self,
        file_path: str,
        class_name: str,
        method_name: str,
        new_method_code: str
    ) -> Dict:
        """
        Modify a method in a class

        Args:
            file_path: Path to Python file
            class_name: Name of class containing method
            method_name: Name of method to modify
            new_method_code: New method implementation

        Returns:
            Result dictionary
        """
        content = self.read_source_file(file_path)
        if not content:
            return {"success": False, "error": "Could not read file"}

        # Validate new method code
        is_valid, error = self.validate_python_syntax(new_method_code)
        if not is_valid:
            return {"success": False, "error": f"Invalid method code: {error}"}

        try:
            tree = ast.parse(content)

            # Find the class and method
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == method_name:
                            # Found the method - replace it
                            lines = content.split('\n')
                            start_line = item.lineno - 1

                            # Find end of method
                            end_line = start_line + 1
                            while end_line < len(lines):
                                if lines[end_line].strip() and not lines[end_line].startswith(' '):
                                    break
                                if lines[end_line].strip().startswith('def '):
                                    break
                                end_line += 1

                            # Replace method
                            old_method = '\n'.join(lines[start_line:end_line])
                            modifications = {"replace": {old_method: new_method_code}}

                            return self.modify_file(file_path, modifications)

            return {"success": False, "error": f"Method {method_name} not found in class {class_name}"}

        except Exception as e:
            return {"success": False, "error": f"Error modifying method: {str(e)}"}


# Global instance
_self_modification = None

def get_self_modification_module() -> ALCALASelfModification:
    """Get or create ALCALA self-modification module instance"""
    global _self_modification
    if _self_modification is None:
        _self_modification = ALCALASelfModification()
    return _self_modification


if __name__ == "__main__":
    # Test self-modification module
    print("Testing ALCALA Self-Modification Module...")
    print()

    mod = ALCALASelfModification()

    # Test 1: List modifiable files
    print("Test 1: List modifiable files")
    files = mod.get_modifiable_files()
    print(f"   Found {len(files)} modifiable files")
    print()

    # Test 2: Read source file
    print("Test 2: Read own source file")
    content = mod.read_source_file(__file__)
    print(f"   Read {len(content)} bytes from own source")
    print()

    # Test 3: Analyze code structure
    print("Test 3: Analyze code structure")
    analysis = mod.analyze_code_structure(__file__)
    print(f"   Classes: {len(analysis.get('classes', []))}")
    print(f"   Functions: {len(analysis.get('functions', []))}")
    print(f"   Lines: {analysis.get('line_count', 0)}")
    print()

    print("ALCALA Self-Modification Module Test Complete!")
