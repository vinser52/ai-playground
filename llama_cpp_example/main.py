from llama_cpp import Llama

def get_llm():
    llm = Llama.from_pretrained(
        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="*Q8_0.gguf",
        verbose=True,
        n_gpu_layers=-1,
    )

    return llm

def run_example():
    llm = get_llm()
    output = llm(
        "Q: Write a simple hello world example in C++ A: ", # Prompt
        max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
        stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
        echo=True # Echo the prompt back in the output
    ) # Generate a completion, can also call create_completion
    
    print("Model output: ")
    print(output)


if __name__ == "__main__":
    run_example()