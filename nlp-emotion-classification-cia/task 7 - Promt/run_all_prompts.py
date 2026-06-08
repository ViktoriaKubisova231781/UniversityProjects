import os
import subprocess
import time

print("===== PROMPT ENGINEERING EXPERIMENT =====")
print("Running all prompt scripts sequentially...")

# Define script paths
script_dir = os.path.dirname(os.path.abspath(__file__))
scripts = [
    os.path.join(script_dir, "prompt_baseline.py"),
    os.path.join(script_dir, "prompt_few_shot.py"),
    os.path.join(script_dir, "prompt_definitions.py"),
    os.path.join(script_dir, "prompt_combined.py"),
    os.path.join(script_dir, "prompt_structured.py")
]

# Run each script
for i, script in enumerate(scripts, 1):
    script_name = os.path.basename(script)
    print(f"\n\n{'='*50}")
    print(f"RUNNING SCRIPT {i}/{len(scripts)}: {script_name}")
    print(f"{'='*50}\n")
    
    # Execute the script
    try:
        subprocess.run(["python", script], check=True)
        print(f"\nSuccessfully completed: {script_name}")
    except subprocess.CalledProcessError:
        print(f"\nError running: {script_name}")
    
    # Add a short delay between scripts
    if i < len(scripts):
        print("\nWaiting 5 seconds before starting next script...")
        time.sleep(5)

print("\n\n===== ALL PROMPT SCRIPTS COMPLETED =====")
print("Now running evaluation script to compare results...")

# Run the evaluation script
eval_script = os.path.join(script_dir, "evaluate_prompts.py")
try:
    subprocess.run(["python", eval_script], check=True)
    print("\nEvaluation completed successfully!")
except subprocess.CalledProcessError:
    print("\nError running evaluation script.")

print("\n===== EXPERIMENT COMPLETE =====")
print("You can now run 'prompt_engineering.py' to create the final script based on the best approach.") 