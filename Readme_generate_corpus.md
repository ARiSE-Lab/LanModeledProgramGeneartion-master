* Copy swdata
*run prepocess.py's genearte_unchanged_change() method 

* Then download Saikat Project
* Change Test.java main method to following

public static void main(String args[]) throws IOException {
		int count = 0;
		//String infile  = "tests/input.java";
		String outFileName = "tests/unchanged_train.txt";
		File dummy = new File(outFileName);
		dummy.delete();
		try (BufferedReader br = new BufferedReader(new FileReader("tests/unchanged_files.txt"))) {
		    String line;
		    while ((line = br.readLine()) != null) {
		    	// process the line.
//		    	String infile = "tests/GPUImageLaplacianFilter.java";
		    	String infile = line;
		    	System.out.println(++count);
//				String outFileName = "tests/unchanged_train.txt";
				if(args.length > 0) 
		           infile = args[0];
				JavaASTTokenizer tokenizer = new JavaASTTokenizer(infile, outFileName);
				tokenizer.tokenize();
				tokenizer.printOutputToFile(); 
		    }
		} 
		
	 }
* In JAVAASTTOKENIZER.JAVA chnage to following:

public void printOutputToFile() throws IOException {
		
		PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(outputFile, true)));
		for(List<Token> sequence : sequenceList){
			
			
			int tokenLength = sequence.size();
			Token []tokenList = new ASTToken[tokenLength];
			tokenList = sequence.toArray(tokenList);
//			for (int i = 0; i < tokenLength; i++){
//				String word = ((ASTToken)tokenList[i]).getContent();
//				if(JavaKeywords.isKeyWord(word)){
//					((ASTToken)tokenList[i]).setType("");
//				}
//				else{
//					if(i < tokenLength -1){
//						String nextWord = ((ASTToken)tokenList[i + 1]).getContent();
//						if (nextWord.trim().compareTo("(")==0 || nextWord.trim().compareTo("()")==0){
//							((ASTToken)tokenList[i]).setType("METHOD_NAME");
//						}
//					}
//				}
//			}
			
			
			writer.print("<METHOD_START> ");
			for (Token token : tokenList){
//				writer.print("\"" + token.text() + "\" , ");
				writer.print( token.text() + " ");
//				System.out.println("\"" + token.text() + "\" , ");
			}
			writer.println(" <METHOD_END>");
			
		}
		writer.close();
//		System.exit(0);
	}
  
 * chnage ASTTOKEN.JAVA to:
 public String text() {
//		StringBuilder sb = new StringBuilder();
//		//sb.append("Text : " + this.text + "\t\tType : " + this.type);
//		//sb.append("<\"" + this.text + "\"");
//		if(this.type != null && this.type.length()!=0){
//			sb.append(this.type );
//		}
//		else{
//			sb.append(this.text);
//		}
//		//sb.append(">");
//		return sb.toString();
		return this.text;
	}
  * run this java eclipe test.java and genearte unchanged_train.txt and python prepocess.py genearte_variable_version_eliminatating_method_tokens(file, output_file)
  
