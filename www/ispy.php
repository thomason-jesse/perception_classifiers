<HTML>

<HEAD>
<TITLE>I, Robot Spy</TITLE>
<META http-equiv="Content-Type" content="text/html; charset=utf-8" />

<SCRIPT TYPE="text/javascript" SRC="http://cdn.robotwebtools.org/EventEmitter2/current/eventemitter2.js"></SCRIPT>
<SCRIPT TYPE="text/javascript" SRC="http://cdn.robotwebtools.org/roslibjs/current/roslib.js"></SCRIPT>

<SCRIPT LANGUAGE="JavaScript">

	// socket vars
	var host = 'ymir.cs.utexas.edu';
	var ros;
	var start_dialog_service;
	var get_say_service;
	var get_point_service;

	// global vars
	var experimental_condition = true;
	var user_id = null;  // supplied from MTurk
	var straight_object_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29];
	var shuffled_object_ids = shuffle(straight_object_ids);
	var all_object_ids = [];
	for (var i=0; i<6; i++)
	{
		var round_object_ids = shuffled_object_ids.slice(i*5, (i*5)+5);
		all_object_ids.push(round_object_ids);
	}
	var object_ids;
	var num_games_played = 0;

	// global instructions
	var user_turn_instructions = "Pick an item and describe it in one phrase to the robot.";
	var robot_turn_instructions = "Click the object you think the robot is describing.";
	
	// what little style exists
	var user_cell_color = 'AliceBlue';
	var system_cell_color = 'GhostWhite';
	var robot_guess_color = 'Gray';
	var human_guess_color = 'Aquamarine';

	function shuffle(o)
	{
	    for(var j, x, i = o.length; i; j = Math.floor(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x);
	    return o;
	}

	function initializeROS()
	{
		ros = new ROSLIB.Ros({url : 'ws://' + host +':9090'});

		start_dialog_service = new ROSLIB.Service({
			ros 		: ros,
			name		: 'start_dialog',
			serviceType	: 'perception_classifiers/startDialog'
		});

		get_say_service = new ROSLIB.Service({
			ros 		: ros,
			name		: 'get_say',
			serviceType	: 'perception_classifiers/getSay'
		});

		get_point_service = new ROSLIB.Service({
			ros 		: ros,
			name		: 'get_point',
			serviceType	: 'perception_classifiers/getPoint'
		});
	}

	function runNewDialogAgent()
	{
		// initialize new dialog agent node by calling startDialog service
		var request = new ROSLIB.ServiceRequest({
			'id': user_id,
			'object_ids': object_ids.join(','),
			'exp_cond': experimental_condition
		});
		start_dialog_service.callService(request, function(result) {});
	}

	// submit ID
	function submitID()
	{
		var start_result = document.getElementById('start_result');
		var id_field = document.getElementById('mturk_id');

		if (id_field.value == "")
			start_result.innerHTML = "Please enter your ID!";
		else
		{
			user_id = id_field.value;
			document.getElementById('start_game').style.display = 'block';
			document.getElementById('enter_id').style.display = 'none';
		}
	}

	// run to show objects and initial instructions (user turn)
	function startGame()
	{
		// hide ask for ID
		document.getElementById('start_game').style.display = 'none';

		// hide continue and history, if they're actually what's up
		document.getElementById('continue_game_block').style.display = 'none';
		document.getElementById('dialog_history_block').style.display = 'none';

		// update task description
		document.getElementById('task_description_text').innerHTML = user_turn_instructions;
		document.getElementById('introduce_task').style.display = 'block';

		// get new objects
		object_ids = all_object_ids[num_games_played];
		object_ids = shuffle(object_ids);

		// clear old dialog on screen, if any
		var table = document.getElementsByName('history')[0];
		while (table.rows.length > 2)
		{
			table.deleteRow(1);
		}

		// launch new agent
		runNewDialogAgent()

		// display objects
		var object_table_row = document.getElementById('object_table_row');
		while (object_table_row.cells.length > 0)
		{
			object_table_row.deleteCell(0);
		}
		for (var idx=0; idx<object_ids.length; idx++)
		{
			var c = object_table_row.insertCell(-1);
			c.innerHTML = "<img src=\"images/"+object_ids[idx]+".JPG\">";
		}
		document.getElementById('object_display').style.display = 'block';

		// show option to begin
		document.getElementById('dialog_start_block').style.display = 'block';
	}

	// run to start a dialog with the system for a new user
	function startDialog()
	{		
		// hide start div and open dialog div
		document.getElementById('dialog_start_block').style.display = 'none';
		document.getElementById('dialog_history_block').style.display = 'block';
		document.getElementById('user_input_box_div').style.display = 'block';
		
		// add robot initial [thinking...] cell
		var table = document.getElementsByName('history')[0];
		var system_row = table.insertRow(table.rows.length-1);
		var system_response_cell = system_row.insertCell(0);
		system_response_cell.innerHTML = "<i>typing...</i>";
		system_response_cell.style.backgroundColor = system_cell_color;
		var system_name_cell = system_row.insertCell(0);
		system_name_cell.innerHTML = "ROBOT";
		system_name_cell.style.backgroundColor = system_cell_color;

		// query for robot say using ROS
		var request = new ROSLIB.ServiceRequest({
			'id': user_id,
		});
		get_say_service.callService(request, function(result) {
			handleSay(result.s);
		});

		return false;
	}

	// run when the user submits a binary response with yes/no buttons
	function userBinary(s)
	{
		document.getElementById('user_input_box').value = s;
		getDialogResponse(document.getElementById('user_input_form'));
	}

	// run when the user submits a new response to the system
	function getDialogResponse(form)
	{
		// get user input and clear text box
		var user_input_raw = form.user_input_box.value;
		form.user_input_box.value = '';
		
		// do initial sanitization of user text and ignore command if it's empty/bogus
		var temp_div = document.createElement("div");
		temp_div.innerHTML = user_input_raw;
		var user_input = temp_div.textContent || temp_div.innerText || "";
		if (user_input.length == 0) return;
		user_input = user_input.toLowerCase();
		user_input = user_input.replace("+", "");
		user_input = user_input.replace("_", "");
		
		// disable user input until system has responded
		document.getElementById('user_input_box_div').style.display = 'none';
		document.getElementById('user_input_binary').style.display = 'none';
		
		// add user text
		var table = document.getElementsByName('history')[0];
		var user_row = table.insertRow(table.rows.length-1);
		var user_response_cell = user_row.insertCell(0);
		user_response_cell.innerHTML = user_input;
		user_response_cell.style.backgroundColor = user_cell_color;
		var user_name_cell = user_row.insertCell(0);
		user_name_cell.innerHTML = "YOU";
		user_name_cell.style.backgroundColor = user_cell_color;

		// add robot initial [thinking...] cell
		addRobotThinkingCell();

		// write user input to file
		getRequest(
		  'write_user_input.php', // URL for the PHP file
		  'user_input='.concat(user_input).concat('&user_id=').concat(user_id), // parameters for PHP
		   invokeWriteUserInputResponse, // handle successful request
		   invokeWriteUserInputError // handle error
		);
	}

	// end the current game and load new objects, if any, or move to survey
	function endGame()
	{
		// load new objects
		num_games_played ++;
		if (num_games_played < all_object_ids.length)
		{
			document.getElementById('continue_game_block').style.display = 'block';
			document.getElementById('user_input_box_div').style.display = 'none';
		}
		else
		{
			// end session and give user code for MTurk
			document.getElementById('end_session_block').style.display = 'block';
			document.getElementsByName('history')[0].deleteRow(-1); // the YOU input row
		}
	}

	function handleSay(s)
	{
		if (s.indexOf("ERROR") > -1)
		{
			document.getElementById('err').innerHTML = s;
			// TODO: if this happens live, need to pay MTurker anyway and have them file report
			return;
		}

		s = s.slice(0, -1);  // cut off trailing '\n'
		s = s.replace('\n', '<br/>');  // reformat for html
		updateSystemCell(s);

		var user_should_point = false;
		var objects_row = document.getElementById('object_table_row');
		if (s.indexOf("I am thinking of an object") > -1)
		{
			var table = document.getElementsByName('history')[0];
			while (table.rows.length > 3)
			{
				table.deleteRow(1);
			}
		}
		if (s.indexOf("I am thinking of an object") > -1 ||
			s.indexOf("That's not the object") > -1)
		{
			document.getElementById('task_description_text').innerHTML = robot_turn_instructions;
			user_should_point = true;
			for (var idx=0; idx < object_ids.length; idx++)
			{
				objects_row.cells[idx].onclick = selectObjectIdxAbstract(idx);
			}
		}

		// if the dialog has concluded
		if (s == "Thanks for playing!")
		{
			endGame();
		}
		else if (user_should_point == false)
		{
			// make yes/no visible if this isn't the prompt question
			if (s.indexOf("Please pick an object") == -1)
			{
				document.getElementById('user_input_binary').style.display = 'block';
			}
		}
	}

	function handlePoint(sel_idx)
	{
		if (sel_idx == -2)  // don't change current pointing behavior
		{
			return;
		}

		var objects_row = document.getElementById('object_table_row');

		// clear robot selections
		for (var idx=0; idx < object_ids.length; idx++)
		{
			objects_row.cells[idx].style.backgroundColor = '';
		}

		// show selection
		if (sel_idx != -1)
		{
			objects_row.cells[sel_idx].style.backgroundColor = robot_guess_color;
		}
		
	}

	function selectObjectIdx(sel_idx)
	{
		var objects_row = document.getElementById('object_table_row');

		// disable clickable objects and clear last selection
		for (var idx=0; idx < object_ids.length; idx++)
		{
			objects_row.cells[idx].onclick = function() {};
			objects_row.cells[idx].style.backgroundColor = '';
		}

		// show selection
		objects_row.cells[sel_idx].style.backgroundColor = human_guess_color;

		// add robot initial [thinking...] cell
		addRobotThinkingCell();

		// write user point to file
		getRequest(
			'write_user_input.php',
			'user_guess='.concat(sel_idx).concat('&user_id=').concat(user_id),
			invokeWriteUserInputResponse,
			invokeWriteUserInputError
		);
	}
	function selectObjectIdxAbstract(sel_idx)
	{
		return function() {selectObjectIdx(sel_idx)};
	}

	function addRobotThinkingCell()
	{
		var table = document.getElementsByName('history')[0];
		var system_row = table.insertRow(table.rows.length-1);
		var system_response_cell = system_row.insertCell(0);
		system_response_cell.innerHTML = "<i>thinking...</i>";
		system_response_cell.style.backgroundColor = system_cell_color;
		var system_name_cell = system_row.insertCell(0);
		system_name_cell.innerHTML = "ROBOT";
		system_name_cell.style.backgroundColor = system_cell_color;
	}

	function updateSystemCell(s)
	{
		var table = document.getElementsByName('history')[0];
		var system_row = table.rows[table.rows.length-2];
		var system_response_cell = system_row.cells[1];
		system_response_cell.innerHTML = s;
	}

	function invokeWriteUserInputResponse(response_text)
	{
		if (response_text.length > 0)
		{
			document.getElementById('err').innerHTML = response_text;
		}

		// query for robot say using ROS
		var request = new ROSLIB.ServiceRequest({
			'id': user_id,
		});
		get_say_service.callService(request, function(result) {
			handleSay(result.s);
		});

		// query for robot point using ROS
		get_point_service.callService(request, function(result) {
			handlePoint(result.oidx);
		});
	}

	function invokeWriteUserInputError()
	{
		document.getElementById('err').innerHTML = "failed to write input";
	}
	
	function endSession()
	{
		document.getElementById('wrap').style.display = 'none';
		document.getElementById('get_code_block').style.display = 'block';
		
		//submit php request
		getRequest(
		  'get_mturk_code.php', // URL for the PHP file
		  'user_id='.concat(user_id), // parameters for PHP
		   submitCodeOutput,  // handle successful request
		   submitCodeError    // handle error
		);
	}
	
	// handles the response, adds the html
	function submitCodeOutput(response_text)
	{
		document.getElementById('get_code_block').innerHTML = response_text;
	}
	
	// handles drawing an error message
	function submitCodeError () {
		document.getElementById('err').innerHTML = "there was an error calling get_mturk_code.php";
	}
	
	// helper function for cross-browser request object
	function getRequest(url, params, success, error)
	{		
		// encode params
		var params_enc = encodeURI(params)
	
		var req = false;
		try{
			// most browsers
			req = new XMLHttpRequest();
		} catch (e){
			// IE
			try{
				req = new ActiveXObject("Msxml2.XMLHTTP");
			} catch (e) {
				// try an older version
				try{
					req = new ActiveXObject("Microsoft.XMLHTTP");
				} catch (e){
					return false;
				}
			}
		}
		if (!req) return false;
		if (typeof success != 'function') success = function () {};
		if (typeof error!= 'function') error = function () {};
		req.onreadystatechange = function(){
			if(req.readyState == 4){
				return req.status === 200 ? 
					success(req.responseText) : error(req.status)
				;
			}
		}
		req.open("POST", url, true);
		req.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
		req.send(params_enc);
		return req;
	}
</SCRIPT>

<STYLE>
#err {
	width=100%
	float:top;
}
#inst {
	width:100%
	float:top;
}
#wrap {
	width:100%;
	margin:0 auto;
}
#wrap_top {
	float:top;
	width:100%;
}
#wrap_bottom {
	float:bottom;
	width:100%;
}

.history_table {
	width:100%;
}
.history_table_speaker {
	width:15%;
}
.history_table_words {
	width:85%;
}
</STYLE>

</HEAD>

<BODY onload="initializeROS()">

<p>
<DIV ID="err" style="display:block;color:red">
</DIV>
</p>

<p>
<DIV ID="enter_id" style="display:block">
	<FORM>
		Enter Amazon Mechanical Turk ID:
		<INPUT ID="mturk_id" TYPE="text" NAME="mturk_id" VALUE="<?=rand(10000,99999)?>" style="width:100%" onkeydown="if (event.keyCode == 13) {document.getElementById('new_dialog_button').click();event.returnValue=false;event.cancel=true;}">
		<INPUT TYPE="button" ID="new_dialog_button" Value="submit" onClick="submitID()" style="display:none">
		<SPAN id="start_result" style="color:red"></SPAN>
	</FORM>
</DIV>
</p>

<p>
<DIV ID="warning" style="display:none;color:red" >
	Do not navigate away from or refresh the page until you have completed all tasks and the exit survey to receive your code. Leaving or refreshing the page <b>will</b> prevent you from completing this HIT.
</DIV>
</p>

<p>
<DIV ID="inst" style="display:none">
</DIV>
</p>

<p>
<DIV ID="wrap">
<DIV ID="wrap_top">
</DIV>
<DIV ID="wrap_bottom">
	<DIV ID="start_game" style="display:none">
		<p>You will play the game I, Spy with a robot. You will take turns using a single sentence to describe an object, then guessing which object is being specified. Then, you will complete a small survey about your experiences and receive your code for Mechanical Turk.</p>
		<p><FORM NAME="start_game_form" ACTION="" METHOD="GET">
			Click the button below to begin.<br/>
			<INPUT TYPE="button" NAME="start_button" Value="See Objects" onClick="startGame()">
		</FORM></p>
	</DIV>

	<DIV ID="introduce_task" style="display:none">
		<b>TASK TO COMPLETE</b><p ID="task_description_text"></p>
	</DIV>

	<DIV ID="object_display" style="display:none">
		<p><table ID="object_table" cellpadding="10" cellspacing="5"><tr ID="object_table_row"></tr></table></p>
	</DIV>
	
	<DIV ID="dialog_start_block" style="display:none">
		<FORM NAME="user_start_dialog_form" ACTION="" METHOD="GET">
			<INPUT TYPE="button" NAME="user_start_dialog_button" Value="Start" onClick="startDialog()">
		</FORM>
	</DIV>

	<DIV ID="dialog_history_block" style="display:none">
		<TABLE NAME="history" style="width:100%" class="history_table">
		<THEAD>
			<TR><TH class="history_table_speaker">&nbsp;</TH>
				<TH class="history_table_words">&nbsp;</TH></TR>
		</THEAD>
		<TBODY>
		<FORM ID="user_input_form" NAME="user_input_form" ACTION="" METHOD="GET">
		<TR ID="user_input_table_row">
			<TD class="history_table_speaker" style="background-color:AliceBlue">YOU</TD>
			<TD class="history_table_words" style="background-color:AliceBlue">
				<DIV ID="user_input_box_div" style="display:block">
					<INPUT TYPE="text" ID="user_input_box" NAME="user_input_box" VALUE="" style="width:100%" onkeydown="if (event.keyCode == 13) {document.getElementsByName('user_input_button')[0].click();event.returnValue=false;event.cancel=true;}">
					<INPUT TYPE="button" NAME="user_input_button" Value="submit" onClick="getDialogResponse(this.form)" style="display:none">
				</DIV>
				<DIV ID="user_input_binary" style="display:none">
					<INPUT TYPE="button" NAME="user_input_binary_yes" Value="Yes" onClick="userBinary('yes')">
					<INPUT TYPE="button" NAME="user_input_binary_no" Value="No" onClick="userBinary('no')">
				</DIV>
			</TD>
		</TR>
		</FORM>
		</TBODY>
		</TABLE>
	</DIV>
	
	<DIV ID="continue_game_block" style="display:none">
		<p><FORM NAME="continue_game_form" ACTION="" METHOD="GET">
			<INPUT TYPE="button" NAME="continue_button" Value="Continue Game" onClick="startGame()">
		</FORM></p>
	</DIV>

	<DIV ID="end_session_block" style="display:none">
		<FORM NAME="user_end_session_form" ACTION="" METHOD="GET">
			<INPUT TYPE="button" NAME="user_end_session_button" Value="Exit and Get Code" onClick="endSession()">
		</FORM>
	</DIV>

</DIV>
</DIV>

<DIV ID="get_code_block" style="display:none">
</DIV>

</p>

</BODY>

</HTML>