import subprocess


bash_script_path = 'scripts/preprocess_data_out.sh'
subprocess.run([bash_script_path], check=True, shell=True)
print("Bash script executed successfully")
