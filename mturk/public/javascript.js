var passedtest = false;

var exams = [
["chair_image1354_box2.jpg", 0],
["chair_image1407_box1.jpg", 0],
["chair_image1492_box1.jpg", 0],
["chair_image1613_box1.jpg", 1],
["chair_image1632_box1.jpg", 0],
["chair_image1643_box1.jpg", 0],
["chair_image599_box1.jpg", 1],
["chair_image1672_box1.jpg", 0],
["chair_image1726_box1.jpg", 0],
["chair_image1759_box1.jpg", 0],
["chair_image6_box1.jpg", 1],
["chair_image1757_box1.jpg", 0],
["chair_image93_box1.jpg", 1]
];

$(document).ready(function()
{
    for (var i = 0; i < exams.length; i++)
    {
        $("#testtable").append("<tr><td><img src='test/ihog/" + exams[i][0] + "'></td><td><input type='radio' id='exam" + i + "yes' name='exam" + i + "'> <label for='exam" + i + "yes'>Yes, it is a chair</label><br><br><input type='radio' id='exam" + i + "no' name='exam" + i + "'> <label for='exam" + i + "no'>No, it is not a chair</label></td></tr>");
    }

    function testuser() 
    {
        window.scrollTo(0, 0);
        $("#test").show();
        $("#submittest").click(function() {
            var score = 0;

            for (var i = 0; i < exams.length; i++)
            {
                var truth = exams[i][1];

                if (truth) {
                    if ($("#exam" + i + "yes").is(":checked")) {
                        score++;
                    }
                }
                else
                {
                    if ($("#exam" + i + "no").is(":checked")) {
                        score++;
                    }
                }
            }

            score = score / exams.length;

            if (score < 0.8)
            {
                alert("You scored less than 80%. Please try again.");
            }
            else
            {
                alert("Congratulations! You scored " + (Math.round(score * 100)) + "%. You may now start the task. You won't have to take this test again.");
                passedtest = true;
                $("#test").hide();
                $("#container").show();
            }
        });
    }

    $("#container").hide();
    $("#instructions").show();

    $("#showinstructions").click(function() {
        $("#container").hide();
        $("#instructions").show();
    });

    $("#start").click(function() {
        if (!mturk_isassigned())
        {
            $("#container").show();
            $("#instructions").hide();
            return;
        }
        server_jobstats(function(data) {
            if (data["newuser"] && !passedtest)
            {
                $("#instructions").hide();
                testuser();
            }
            else
            {
                $("#container").show();
                $("#instructions").hide();
            }
        });
    });

    var showing = 0;

    if (!mturk_isassigned())
    {
        mturk_acceptfirst();
    }
    else
    {
        mturk_showstatistics();
    }

    var parameters = mturk_parameters();
    if (!parameters["id"]) 
    {
        $("body").html("Missing ID query string.");
        return;
    }

    var marks = {};

    server_request("getjob", [parameters["id"]], function(data) {
        $(".category").html(data["category"]);

        var urls = [];
        for (var i = 0; i < data["windows"].length; i++)
        {
            urls.push("/images/" + data["windows"][i][1]);
        }
        preload(urls, function(p) { $("#debug").html(p); } );

        var lastimts = 0;

        $("#nextim").click(function() {
            if ((new Date()).getTime() - lastimts < 300)
            {
                alert("You are going too fast. Please look at the image again.");
                return;
            }
            if (marks[showing] == undefined)
            {
                alert("Please make a choice before proceeding.");
                return;
            }
            if (showing == data["windows"].length-1) return;
            $("#window" + showing).hide();
            showing++;
            update();

            if (!mturk_isassigned()) {
                $("#turkic_acceptfirst").hide();
                window.setTimeout(function() {
                    $("#turkic_acceptfirst").show();
                    window.setTimeout(function() {
                        $("#turkic_acceptfirst").hide();
                        window.setTimeout(function() {
                            $("#turkic_acceptfirst").show();
                        }, 200);
                    }, 200);
                }, 200);
            }

            lastimts = (new Date()).getTime();
        });
        $("#previm").click(function() {
            if (showing == 0) return;
            $("#window" + showing).hide();
            showing--;
            update();
        });

        $("#doesnotcontain").click(function() {
            marks[showing] = -1;
        });
        $("#doescontain").click(function() {
            marks[showing] = 1;
        });

        $("#submit").click(function() {
            var counter = 0;
            var payload = "[";
            for (var i in marks) {
                payload += "[" + data["windows"][i][0] + "," + marks[i] + "],";
                counter++;
            }
            payload = payload.substr(0, payload.length - 1) + "]";

            if (counter != data["windows"].length)
            {
                alert("You must have made a decision for every image before you can submit.");
                return;
            }

            mturk_submit(function(redirect) {
                server_post("savejob", [parameters["id"]], payload, function(data) {
                    redirect();
                });
            });
        });

        $(window).keypress(function(e) {
            if (e.which == 121) {
                $("#doescontain").click();
                $("#nextim").click();
            }
            if (e.which == 110) {
                $("#doesnotcontain").click();
                $("#nextim").click();
            }
        });

        function showwindow(i)
        {
            $("#windows").html("<img src='/images/" + data["windows"][i][1] + "'>");
        }

        function update()
        {
            showwindow(showing);
            $("#status").html("Showing image " + (showing+1) + " of " + data["windows"].length);

            if (marks[showing] == 1)
            {
                $("#doescontain").attr("checked", true);
            }
            else if (marks[showing] == -1)
            {
                $("#doesnotcontain").attr("checked", true);
            }
            else
            {
                $("#doesnotcontain").attr("checked", false);
                $("#doescontain").attr("checked", false);
            }
        }

        update();
    });
});
