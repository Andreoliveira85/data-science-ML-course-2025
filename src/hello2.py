def greet_chef(name):
    """Function to greet a chef in the kitchen"""
    return f"👋 Hello, Chef {name}! Welcome to the kitchen!"

def prepare_workspace():
    """Prepare the kitchen workspace"""
    print("🧹 Cleaning the kitchen...")
    print("🔪 Sharpening knives...")
    print("🍽️ Setting up plates...")
    print("✅ Kitchen is ready for cooking!")

def main():
    """Main function - what happens when we run this recipe book directly"""
    print("📖 Opening the recipe book...")
    
    # Test our greeting function
    message = greet_chef("Data Scientist")
    print(message)
    
    # Prepare kitchen
    prepare_workspace()
    
    print("🍳 Ready to start cooking!")

# This only runs when we're testing THIS recipe book directly
if __name__ == "__main__":
    print("🚀 Testing the kitchen setup recipe...")
    main()
    print("✅ Kitchen setup recipe works perfectly!")