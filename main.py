import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from src.ingest import IngestionEngine
from src.vector_store import VectorStoreManager
from src.llm import LLMInterface
from src.rag_engine import RAGOrchestrator

console = Console()

@click.group()
def cli():
    """
    ðŸ§  NeuroSearch: Modular RAG Engine CLI.
    """
    pass

@cli.command()
@click.option("--dir", default="./data/docs", help="Directory containing documents to ingest.")
def ingest(dir):
    """
    Ingest documents into the vector store.
    """
    console.print(Panel(f"ðŸš€ Ingesting documents from: [bold cyan]{dir}[/bold cyan]", title="NeuroSearch Ingestion"))
    
    # Initialize components
    ingestion_engine = IngestionEngine()
    vector_store_manager = VectorStoreManager()
    
    try:
        # Load and split documents
        docs = ingestion_engine.load_documents(dir)
        if not docs:
            console.print("[yellow]No documents found.[/yellow]")
            return

        chunks = ingestion_engine.split_documents(docs)
        
        # Add to vector store
        vector_store_manager.add_documents(chunks)
        
        console.print(f"[green]Successfully ingested {len(chunks)} chunks into the vector store![/green]")
        
    except Exception as e:
        console.print(f"[red]Error during ingestion: {e}[/red]")

@cli.command()
@click.option("--model", default="llama3", help="Ollama model to use.")
def chat(model):
    """
    Start an interactive chat session with your documents.
    """
    console.print(Panel(f"ðŸ’¬ Starting chat session with model: [bold green]{model}[/bold green]", title="NeuroSearch Chat"))
    
    # Initialize components
    vector_store_manager = VectorStoreManager()
    llm_interface = LLMInterface(model_name=model)
    rag_orchestrator = RAGOrchestrator(vector_store_manager, llm_interface)
    
    console.print("Type 'exit' or 'quit' to end the session.\n")
    
    while True:
        user_query = Prompt.ask("[bold blue]You[/bold blue]")
        
        if user_query.lower() in ["exit", "quit"]:
            console.print("[yellow]Goodbye![/yellow]")
            break
            
        with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
            try:
                response = rag_orchestrator.query(user_query)
                answer = response.get("result", "No answer found.")
                sources = response.get("source_documents", [])
                
                console.print(f"\n[bold green]NeuroSearch AI:[/bold green] {answer}\n")
                
                if sources:
                    console.print("[italic dim]Sources:[/italic dim]")
                    for doc in sources[:2]: # Show top 2 sources
                        source_name = doc.metadata.get('source', 'Unknown')
                        console.print(f"- [dim]{source_name}[/dim]")
                console.print("\n" + "-"*50 + "\n")
                
            except Exception as e:
                console.print(f"[red]Error generating response: {e}[/red]")

if __name__ == "__main__":
    cli()
