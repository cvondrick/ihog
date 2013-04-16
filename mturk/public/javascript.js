var passedtest = false;

var exams = [
["chair_image1354_box2.jpg", 0],
["chair_image1407_box1.jpg", 0],
["chair_image1492_box1.jpg", 0],
["chair_image1613_box1.jpg", 1],
["chair_image1643_box1.jpg", 0],
["chair_image599_box1.jpg", 1],
["chair_image1672_box1.jpg", 0],
["chair_image1726_box1.jpg", 0],
["chair_image1759_box1.jpg", 0],
["chair_image6_box1.jpg", 1],
["chair_image93_box1.jpg", 1]
];

//var exams = [
//["person_image52_box1.jpg", 1],
//["person_image53_box1.jpg", 1],
//["person_image59_box1.jpg", 0],
//["person_image67_box1.jpg", 1],
//["person_image71_box3.jpg", 0],
//["person_image72_box1.jpg", 1],
//["person_image73_box7.jpg", 1],
//["person_image81_box1.jpg", 0],
//["person_image95_box3.jpg", 0],
//["person_image96_box5.jpg", 0],
//];
//
//var exams = [
//["cat_image4408_box1.jpg", 0],
//["cat_image4410_box1.jpg", 1],
//["cat_image4473_box1.jpg", 0],
//["cat_image4515_box1.jpg", 0],
//["cat_image4515_box2.jpg", 0],
//["cat_image4605_box1.jpg", 0],
//["cat_image4611_box2.jpg", 1],
//["cat_image4657_box1.jpg", 1],
//["cat_image4658_box1.jpg", 0],
//["cat_image4717_box1.jpg", 1],
//["cat_image4738_box1.jpg", 0],
//];

//var exams = [
//["car_image123_box1.jpg", 1],
//["car_image156_box5.jpg", 1],
//["car_image203_box1.jpg", 0],
//["car_image20_box2.jpg", 0],
//["car_image213_box3.jpg", 0],
//["car_image235_box1.jpg", 0],
//["car_image262_box1.jpg", 0],
//["car_image316_box2.jpg", 0],
//["car_image327_box2.jpg", 0],
//["car_image332_box2.jpg", 0],
//["car_image356_box1.jpg", 0],
//["car_image359_box4.jpg", 0],
//["car_image370_box1.jpg", 0],
//["car_image43_box1.jpg", 1],
//["car_image4_box1.jpg", 1],
//["car_image72_box1.jpg", 0],
//];
//
//var exams = [
//["horse_image13_box1.jpg", 1],
//["horse_image30_box1.jpg", 0],
//["horse_image14_box1.jpg", 1],
//["horse_image32_box1.jpg", 0],
//["horse_image4919_box1.jpg", 1],
//["horse_image4948_box2.jpg", 0],
//["horse_image56_box1.jpg", 0],
//["horse_image82_box3.jpg", 1],
//["horse_image84_box1.jpg", 1],
//["horse_image87_box5.jpg", 0],
//];
//
//var examples = [
//["car_image38_box1.jpg", "Yes"],
//["car_image43_box1.jpg", "Yes"],
//["car_image4_box3.jpg",  "Yes"],
//["car_image69_box1.jpg", "Yes"],
//["car_image168_box3.jpg", "Yes (pickups are cars)"],
//["car_image262_box1.jpg", "No (it's a bus)"],
//["car_image69_box3.jpg", "No"],
//["car_image72_box1.jpg", "No"],
//["car_image80_box2.jpg", "No"],
//];
//
//var examples = [
//["horse_image13_box1.jpg", "Yes"],
//["horse_image160_box1.jpg", "Yes"],
//["horse_image175_box2.jpg", "No"],
//["horse_image181_box1.jpg", "No (it's a dog)"],
//["horse_image209_box1.jpg", "Yes"],
//["horse_image228_box1.jpg", "Yes"],
//["horse_image252_box2.jpg", "No"],
//["horse_image40_box2.jpg", "No"],
//["horse_image84_box1.jpg", "Yes"],
//["horse_image9_box1.jpg", "No (it's a sheep)"],
//];

//var examples = [
//["person_image10_box1.jpg", "Yes"],
//["person_image14_box4.jpg", "Yes"],
//["person_image30_box2.jpg", "Yes"],
//["person_image40_box2.jpg", "Yes"],
//["person_image41_box2.jpg", "No"],
//["person_image51_box2.jpg", "No"],
//["person_image56_box2.jpg", "No"],
//["person_image67_box2.jpg", "Yes"],
//["person_image73_box5.jpg", "Yes"],
//["person_image78_box1.jpg", "No"],
//["person_image4908_box1.jpg", "No"],
//["person_image4914_box1.jpg", "No"],
//["person_image4916_box1.jpg", "No"],
//["person_image4926_box7.jpg", "Yes"],
//["person_image4930_box1.jpg", "No"],
//];
//
//var examples = [
//["cat_image16_box1.jpg", "Yes"],
//["cat_image1_box1.jpg", "No"],
//["cat_image204_box1.jpg", "Yes"],
//["cat_image204_box3.jpg", "Yes"],
//["cat_image273_box1.jpg", "Yes"],
//["cat_image306_box1.jpg", "No"],
//["cat_image338_box2.jpg", "No"],
//["cat_image369_box1.jpg", "No (its a dog)"],
//["cat_image415_box1.jpg", "No"],
//["cat_image428_box1.jpg", "No"],
//["cat_image435_box1.jpg", "No"],
//["cat_image64_box1.jpg", "Yes"],
//];
var examples = [
["chair_image102_box1.jpg", "No"],
["chair_image1518_box1.jpg", "Yes"],
["chair_image157_box3.jpg", "No (its a sofa)"],
["chair_image184_box1.jpg", "Yes"],
["chair_image45_box1.jpg", "No"],
["chair_image476_box1.jpg", "Yes"],
["chair_image599_box1.jpg", "Yes"],
["chair_image98_box1.jpg", "Yes"],
];


$(document).ready(function()
{
    for (var i = 0; i < exams.length; i++)
    {
        $("#testtable").append("<tr><td style='padding:20px;text-align:center;'><img src='test/original/" + exams[i][0] + "'></td><td><input type='radio' id='exam" + i + "yes' name='exam" + i + "'> <label for='exam" + i + "yes'>Yes, it is a chair</label><br><br><input type='radio' id='exam" + i + "no' name='exam" + i + "'> <label for='exam" + i + "no'>No, it is not a chair</label></td></tr>");
    }

    function buildexamples()
    {
        for (var i = 0; i < examples.length; i++)
        {
            $("#examples").append("<tr><td style='text-align:center;'><img width='200' style='padding:10px;' src='examples/original/" + examples[i][0] + "'></td><td style='text-align:center;'>" + examples[i][1] + "</td></tr>");
        }
    }

    buildexamples();

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

            if (score < 0.90)
            {
                alert("You scored " + (Math.round(score * 100)) + "%. You need 90% to pass. Please try again.");
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

    if (false)
    {
        $("#instructions").hide();
        testuser();
    }

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
            if ((new Date()).getTime() - lastimts < 100)
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
            $("#windows").html("<img src='/images/" + data["windows"][i][1] + "' height='300'>");
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
