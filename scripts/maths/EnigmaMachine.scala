// Virtual reconstruction of what the enigma machines did
// The replication is of the improved version where the input could result in the same output
// mark, 2013

object EnigmaMachine {
//////////////// ENCRYPTION/DECRYPTION FUNCTIONS \\\\\\\\\\\\\\\\\\\\\
    //to replicate the simple machine circuit chars are converter to 0 - 25
    def convertMessageToSimpleInt(rawInput: String):List[Int] = {
        val userMessage = rawInput.toUpperCase
      (for (x <- userMessage) yield if (x != 32) (x.toInt - 65) else 26).toList
    }
    
    //for all integers in the list, convert
    def convertMessageToString(input: List[Int]): String = {
      (for (x <- input) yield if (x == 26) " " else (x + 65).toChar).mkString
    } 
    
    def enigmaGo(rotorChoices: List[Int], rotorPositions: List[Int], inputMessage: List[Int], encrypt: Boolean): List[Int] = {    
      // incrementing the rotors is dont based on the number of previous chars.. instead of the previous position
      def incrementRots(nth: Int): List[Int] = {
        val rot0 = (rotorPositions(0) + nth) % 26 //fast rotor
        val rot1 = (rotorPositions(1) + (nth / 26)) % 26 //medium rotor
        val rot2 = (rotorPositions(2) + ((nth / 26) /26)) % 26 //slow rotor
        List(rot0, rot1, rot2)
      }
      //applies the rotor encryption then the scramble board
      def transmuteElement(element: Int, rotPosApp: List[Int]): Int = {
        val rotorTotal = rotorList(rotorChoices(0))(rotPosApp(0)) +
              rotorList(rotorChoices(1))(rotPosApp(1)) +
              rotorList(rotorChoices(2))(rotPosApp(2)) 
        
      def scramble(inputVal:Int):Int = plugMap1.find(x => (x._1 == inputVal) || (x._2 == inputVal)) match {
            case Some(y) => if (y._1 == inputVal) y._2 else y._1 
          case None => inputVal
      }
        if (element == 26) element
        else if (encrypt) (scramble(element) + rotorTotal) % 26 else { 
            //decrypt is a bit messy atm :( 
            if (element - (rotorTotal % 26) < 0) scramble(element - (rotorTotal % 26) + 26) else scramble(element - (rotorTotal % 26))
          }     
      }
      
      //Accumulator hold the input and output lists, when input list is empty, return output.. also tail recursive
      def enigmaAccu(remInput: List[Int], rotPos: List[Int], accumOutput: List[Int]): List[Int] = {
              if (remInput.isEmpty) accumOutput
            else enigmaAccu(remInput.tail, incrementRots(accumOutput.length), accumOutput:+(transmuteElement(remInput.head, rotPos)))
      }
      
      enigmaAccu(inputMessage, rotorPositions, Nil) //start the process
    }
////////////////////   \\\\\\\\\\\\\\\\\\\\\\\\\
    
//////////////// UI STUFF \\\\\\\\\\\\\\\\\\\\\
    def useMachine(rotorChoices: List[Int], rotorStartVals: List[Int], activity: String) = {
            spamLine
            val inputMessage = convertMessageToSimpleInt(askInput("Please enter a message to be " + activity + ", use only a-z: "))
            val resultingMessage = enigmaGo(rotorChoices, rotorStartVals, inputMessage, if (activity == "encrypted") true else false)
            spamLine
            println("Your message has been " + activity + ": ")
        println(convertMessageToString(resultingMessage))
            spamLine
            exit(0)
    }

    def getRotorChoices(s1:String, p: String => Boolean):List[Int] = {
        val slot0 = getUserInput(s1 + "fast slot: ", "Entry Invalid.", p).mkString
        val slot1 = getUserInput(s1 + "medium slot: ", "Entry Invalid.", p).mkString
        val slot2 = getUserInput(s1 + "slow slot: ", "Entry Invalid.", p).mkString
        List(slot0.toInt,slot1.toInt,slot2.toInt)
    }
    
    def spamLine = println("##############################################")
    
    def askInput(statement: String) = { 
      val input = readLine(statement)
      if (input == "") "99" else input
    }
    
    def askAgain(statement: String, f: => String) = {
        println(statement)
        f
    } 
    
    def inputVals(f1: => String, f2: => String) = Stream.cons( f1, Stream.continually(f2))
    
    def getUserInput(s1: String, s2: String, p:String => Boolean) = inputVals(askInput(s1), askAgain(s2, askInput(s1))).find(p)
////////////////////   \\\\\\\\\\\\\\\\\\\\\\\\\
    
//////////////// MAIN \\\\\\\\\\\\\\\\\\\\\ 
        def main(args: Array[String]) {
            println("### WELCOME TO THE ENIGMA MACHINE! ###" )
            // GET INITIAL MACHINE SETTINGS FROM USER 
            // ROTOR PLACEMENT CONFIG
            val rotorChoices = getRotorChoices("Choose rotor (0 - 4) for ", x => (x.toInt >= 0 && x.toInt < 5)) 
            // ROTOR START VALS 
            val rotorStartVals =  getRotorChoices("Choose rotor START VAL (0 - 25) for ", x => (x.toInt >= 0 && x.toInt < 25))
            // PLUG MAP
            println("There is only 1 plug map option at present:") 
            println(plugMap1.toString.drop(4))
            //ENCRYPT OF DECRYPT - ONLY RELEVENT FOR GUI TEXT.. bleh not quite need to reverse circuit manually
            val userChoice = getUserInput("Press 1 to encrypt, 2 to decrypt: ", "Entry Invalid.", x => (x == "2" || x == "1"))
            userChoice.mkString match {
              case "1" => useMachine(rotorChoices, rotorStartVals, "encrypted") 
              case "2" => useMachine(rotorChoices, rotorStartVals, "decrypted")
              case default => exit(0)
            }
    }
////////////////////   \\\\\\\\\\\\\\\\\\\\\\\\\

//////////////// CONSTANTS \\\\\\\\\\\\\\\\\\\\\
    val rotorList = List(List(6, 17, 4, 3, 23, 22, 21, 5, 0, 19, 20, 11, 16, 9, 13, 24, 18, 1, 8, 2, 7, 10, 12, 14, 25, 15)
    ,List(2, 7, 23, 12, 13, 15, 18, 1, 8, 11, 0, 3, 10, 5, 24, 4, 17, 19, 6, 9, 20, 14, 25, 21, 16, 22)
    ,List(23, 1, 19, 18, 0, 10, 3, 8, 9, 21, 2, 12, 13, 14, 24, 6, 20, 16, 11, 17, 15, 7, 4, 25, 22, 5)
    ,List(21, 22, 16, 2, 18, 20, 15, 9, 5, 1, 17, 10, 3, 6, 0, 13, 11, 23, 4, 12, 19, 8, 7, 25, 24, 14)
    ,List(25, 3, 11, 7, 5, 15, 24, 10, 21, 6, 12, 17, 8, 23, 2, 18, 19, 14, 4, 0, 20, 9, 22, 1, 16, 13))
    
    val plugMap1 = List((3,7),(11,16),(18,24),(17,9),(21,19),(14,13),(0,12),(1,25),(22,20),(4,6))
////////////////////   \\\\\\\\\\\\\\\\\\\\\\\\\
}