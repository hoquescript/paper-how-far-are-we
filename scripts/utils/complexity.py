import csv
import os
import shutil
import subprocess
import scripts.report.complexity as complexity

METRICS = [
    # --- Size Metrics (Volume & Density) ---
    "CountLine",                # Total lines
    "CountLineCode",            # Code lines
    "CountLineComment",         # Comment lines
    "CountLineBlank",           # Blank lines
    "CountStmt",                # Total statements
    "CountStmtDecl",            # Declarative statements
    "CountStmtExe",             # Executable statements
    "CountLineCodeDecl",        # Lines of code with declarations
    "CountLineCodeExe",         # Lines of code with execution

    # --- Average Size Metrics (Consistency) ---
    "AvgLine", 
    "AvgLineCode", 
    "AvgLineComment", 
    "AvgLineBlank",

    # --- Declaration Metrics (Structure Count) ---
    "CountDeclClass",           # Number of classes
    "CountDeclFunction",        # Number of functions
    "CountDeclExecutableUnit",  # Number of executable units

    # --- Complexity Metrics (Control Flow) ---
    "Cyclomatic",               # Cyclomatic complexity
    "MaxCyclomatic",            # Max complexity in file
    "AvgCyclomatic",            # Average complexity
    "SumCyclomatic",            # Total complexity
    "MaxNesting",               # Deepest nesting level

    # --- Documentation Metrics (Readability) ---
    "RatioCommentToCode",       # Ratio of comments to code

    # --- Halstead Metrics (Lexical Entropy/Vocabulary) ---
    "HalsteadEffort",                # Halstead Effort
    "HalsteadDifficulty",            # Halstead Difficulty

    # --- Architecture/OO Metrics (Coupling & Inheritance) ---
    "CountClassCoupled",        # Coupling (Dependencies)
    "MaxInheritanceTree"        # Inheritance Depth
]


def getHalsteadVolume(report):
    effort_str = str(report.get("HalsteadEffort", "0"))
    difficulty_str = str(report.get("HalsteadDifficulty", "0"))

    try:
        effort = float(effort_str.replace(",", ""))
        difficulty = float(difficulty_str.replace(",", ""))
        
        # Avoid division by zero
        if difficulty == 0:
            return 0.0
            
        volume = effort / difficulty
        return volume

    except ValueError:
        return 0.0

def parse_metrics(project_path):
    db = complexity.open(project_path)

    with open ("understand.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=['Name', 'HalsteadVolume', *METRICS])
        writer.writeheader()
        for entity in db.ents("File"):
            if entity.library():
                continue
            report = entity.metric(METRICS)

            # 2. Aggregation Logic: Look for classes INSIDE this file
            # "Define" reference tells us which classes are created in this file
            classes_in_file = file.refs("Define", "Class")

            max_dit = 0
            max_cbo = 0

            for ref in classes_in_file:
                class_entity = ref.ent()
                
                # Fetch class-specific metrics
                class_metrics = class_entity.metric(["MaxInheritanceTree", "CountClassCoupled"])
                
                # Extract values safely (handle None)
                dit = class_metrics.get("MaxInheritanceTree", 0)
                cbo = class_metrics.get("CountClassCoupled", 0)

                # Logic: We report the WORST case (Max) for the file
                # You could also use sum() if you prefer total complexity
                if dit and dit > max_dit:
                    max_dit = dit
                if cbo and cbo > max_cbo:
                    max_cbo = cbo

            # 3. Add calculated class metrics to the file report
            report["MaxInheritanceTree"] = max_dit
            report["CountClassCoupled"] = max_cbo
            
            # 4. Calculate Halstead Volume
            report["HalsteadVolume"] = getHalsteadVolume(report)
            report["Name"] = file.longname()

            writer.writerow(report)

    db.close()

if __name__ == "__main__":
    # Executable Software
    understand_exe = "/Applications/Understand.app/Contents/MacOS/und"    

    # Source Code
    source_path = os.path.abspath("temp/")

    # Check what files are in temp/
    if os.path.exists(source_path):
        files = os.listdir(source_path)
    
    # Understand project path
    project_path = os.path.abspath(".analysis.und")

    # Cleanup
    if os.path.exists(project_path):
        shutil.rmtree(project_path)

    # Running the command to create a project
    subprocess.run([understand_exe, 
                    "create", 
                    "-languages", "Python",
                    project_path])

    # Referencing the source code directory to the tools
    subprocess.run([understand_exe, "add", source_path, project_path])

    # Analyze the project
    result = subprocess.run([understand_exe, "analyze", project_path], capture_output=True)

    # Parsing the metrics
    parse_metrics(project_path)
