' PING-IP-to-Excel 
' IP to Hostname --- or --- Hostname to IP 
'   
' Inspired by: Torgeir Bakken (MVP)    
' http://groups.google.com/group/microsoft.public.scripting.vbscript/msg/a465907f8dc6e265?pli=1 
'  
' Author: Angel Cruz, angelcruzpr@live.com 
' Script to get a list of hostnames or IP's from a texfile and return an Excel Worksheet with 
' 
' HOSTNAME  IP  RESULT  LATENCY 
' --------  --  ------  ------- 
' 
' Using only PING at the command prompt 
' 
' Version 7.0 Jan/16/2011 
' 
Dim strHostname, strIP, strPingResult, IntLatency 
 
intRow = 2 
Set objExcel = CreateObject("Excel.Application") 
 
With objExcel 
     
    .Visible = True 
    .Workbooks.Add 
     
    .Cells(1, 1).Value = "XXXXXXXXXXXXXXXXXXXXXXXXXXX" 
    .Cells(1, 2).Value = "XXXXXXXXXXXXXX" 
    .Cells(1, 3).Value = "XXXXXXX" 
    .Cells(1, 4).Value = "XXXXXXX" 
     
    .Range("A1:D1").Select 
    .Cells.EntireColumn.AutoFit 
     
    .Cells(1, 1).Value = "Hostname" 
    .Cells(1, 2).Value = "IP" 
    .Cells(1, 3).Value = "Result" 
    .Cells(1, 4).Value = "Latency" 
     
End With  
 
'--- Input Text File with either Hostames or IP's --- 
Set Fso = CreateObject("Scripting.FileSystemObject") 
Set InputFile = fso.OpenTextFile("machines.txt") 
 
Do While Not (InputFile.atEndOfStream) 
     
    strHostname = InputFile.ReadLine 
     
    Set WshShell = WScript.CreateObject("WScript.Shell") 
     
    Call PINGlookup( strHostname, strIP, strPingResult, intLatency ) 
     
    With objExcel 
        .Cells(intRow, 1).Value = strHostname 
        .Cells(intRow, 2).Value = strIP 
        .Cells(intRow, 3).Value = strPingResult 
        .Cells(intRow, 4).Value = intLatency 
    End With 
     
    intRow = intRow + 1 
     
Loop 
 
With objExcel 
    .Range("A1:D1").Select 
    .Selection.Interior.ColorIndex = 19 
    .Selection.Font.ColorIndex = 11 
    .Selection.Font.Bold = True 
    .Cells.EntireColumn.AutoFit 
End With 
 
 
'------------- Subrutines and Functions ---------------- 
 
Sub PINGlookup(ByRef strHostname, ByRef strIP, ByRef strPingResult, ByRef intLatency )  
    ' Both IP address and DNS name is allowed  
    ' Function will return the opposite  
     
    ' Check if the Hostname is an IP 
    Set oRE = New RegExp  
    oRE.Pattern = "^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$"  
     
    ' Sort out if IP or Hostname 
    strMachine = strHostname 
    bIsIP = oRE.Test(strMachine)  
    If bIsIP Then  
        strIP = strMachine 
        strHostname = "-------" 
    Else 
        strIP = "-------" 
        strHostname = strMachine 
    End If  
     
     
    ' Get a temp filename and open it 
    Set osShell = CreateObject("Wscript.Shell") 
    Set oFS = CreateObject("Scripting.FileSystemObject")  
    sTemp = osShell.ExpandEnvironmentStrings("%TEMP%")  
    sTempFile = sTemp & "\" & oFS.GetTempName  
     
    ' PING and check if the IP exists 
    intT1 = Fix( Timer * 1000 )  
    osShell.Run "%ComSpec% /c ping -a " & strMachine & " -n 1 > " & sTempFile, 0, True 
    intT2 = Fix( Timer * 1000 )  
    intLatency = Fix( intT2 - intT1 ) / 1000 
     
     
    ' Open the temp Text File and Read out the Data  
    Set oTF = oFS.OpenTextFile(sTempFile)  
     
    ' Parse the temp text file  
    strPingResult = "-------" 'assume failed unless... 
    Do While Not oTF.AtEndoFStream  
         
        strLine = Trim(oTF.Readline)  
        If strLine = "" Then  
            strFirstWord = "" 
        Else  
            arrStringLine = Split(strLine, " ", -1, 1) 
            strFirstWord = arrStringLine(0) 
        End If  
         
        Select Case strFirstWord 
             
            Case "Pinging"  
                If arrStringLine(2) = "with" Then 
                    strPingResult = "-------" 
                    strHostname = "-------" 
                Else 
                    strHostname = arrStringLine(1) 
                    strIP = arrStringLine(2) 
                    strLen = Len( strIP ) - 2 
                    strIP = Mid( strIP, 2, strLen ) 
                    strPingResult = "Ok" 
                End If  
                Exit Do             
            'End Case 
             
            Case "Ping" ' pinging non existent hostname 
                strPingResult = "------" 
                Exit Do     
            'End Case  
                 
        End Select 
         
    Loop  
     
    'Close it  
    oTF.Close  
    'Delete It  
    oFS.DeleteFile sTempFile  
     
     
End Sub  