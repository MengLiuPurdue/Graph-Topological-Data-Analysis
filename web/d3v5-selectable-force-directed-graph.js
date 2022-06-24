const offset = 20;

var curve = d3.line()
    .curve(d3.curveCardinal.tension(0.85));
function drawCluster(d) {
    return curve(d.path);
}      

class MySet extends Set{
    add(elem){
      return super.add(typeof elem === 'object' ? JSON.stringify(elem) : elem);
    }
    has(elem){
      return super.has(typeof elem === 'object' ? JSON.stringify(elem) : elem);
    }
}

function create_hulls(groups,expands,offset) {
    // create point sets
    var hulls = [];
    for (var i = 0; i < Object.keys(groups).length; i++) {
        var l = [];
        if (expands[i] == false){
            continue;
        }
        for (var k = 0; k < groups[i].length; k++) {
            var n = groups[i][k];
            l.push([n.x-offset, n.y-offset]);
            l.push([n.x-offset, n.y+offset]);
            l.push([n.x+offset, n.y-offset]);
            l.push([n.x+offset, n.y+offset]);
        }   
        hulls.push({"group":[i], "path":d3.polygonHull(l)});
    }
  
    // create convex hulls
    return hulls;
  }

function update_network(data, init_nodes, graph, expand, groups, is_initial, dblclick_id, group_meta, component_init_pos, selected_components) {
    var nodes = [], links = [], cnt = 0, node_map = {}, prev_locs = {};
    const n = data.nodes.length;
    let link_set = new MySet();
    if (is_initial == false) {
        graph.nodes.forEach(function(d) {
            prev_locs[d.id] = [d.x,d.y];
        });
    }
    // console.log(prev_locs);
    for (var i=0; i<Object.keys(expand).length; i++){
        if (selected_components[groups[i][0].cid] == false) continue;
        if (expand[i]) {
            for (var j=0; j<groups[i].length; j++){
                var node = groups[i][j];
                node.size = 1;
                node.node_type = "node";
                // console.log(node);
                // if newly expanded, set initial position to be previous cluster centroid
                if (dblclick_id == n+i) {
                    node.x = prev_locs[dblclick_id][0];
                    node.y = prev_locs[dblclick_id][1];
                }
                // console.log(node);
                if (!(node.id in node_map)){
                    node["pieChart"] = [{"color":node.prediction+1,"percent":100}];
                    nodes.push(node);
                    node_map[node.id] = cnt;
                    cnt += 1;
                }
            }
            // console.log(nodes);
            // console.log(groups[i]);
        }else{
            var node = {"id":n+i,"group":[i],"pieChart":group_meta[i],"size":groups[i].length,"node_type":"copmponent","cid":groups[i][0].cid};
            if (node.id in prev_locs) {
                node.x = prev_locs[node.id][0];
                node.y = prev_locs[node.id][1];
            }else{
                node.x = component_init_pos[groups[i][0].cid][0];
                node.y = component_init_pos[groups[i][0].cid][1];
            }
            // if collapsed a cluster, set initial position to be mean x/y
            // console.log(node.group);
            // console.log(prev_locs);
            // console.log(groups[node.group[0]]);
            if (n+i == dblclick_id) {
                var all_x = 0, all_y = 0;
                groups[node.group[0]].forEach(function(d){
                    all_x += prev_locs[d.id][0];
                    all_y += prev_locs[d.id][1];
                });
                node.x = all_x/groups[i].length;
                node.y = all_y/groups[i].length;
            }
            nodes.push(node);
            node_map[n+i] = cnt;
            cnt += 1;
        }
    }
    for (var i=0; i < data.links.length; i++){
        var gsi = init_nodes[data.links[i].source].group;
        if (selected_components[groups[gsi[0]][0].cid] == false) continue;
        var gsj = init_nodes[data.links[i].target].group;
        var ni = init_nodes[data.links[i].source].id;
        var nj = init_nodes[data.links[i].target].id;
        gsi.forEach(function(gi){
            gsj.forEach(function(gj){
                var ei = gi+n;
                var ej = gj+n;
                if (!expand[gi] & !expand[gj]){
                   if (link_set.has([ej,ei]) == false){
                       link_set.add([ei,ej]);
                   } 
                }
                if (expand[gi] & expand[gj]){
                    if (link_set.has([nj,ni]) == false){
                        link_set.add([ni,nj]);
                    } 
                }
                if (!expand[gi] & expand[gj]){
                    if (link_set.has([nj,ei]) == false){
                        link_set.add([ei,nj]);
                    } 
                }
                if (expand[gi] & !expand[gj]){
                    if (link_set.has([ej,ni]) == false){
                        link_set.add([ni,ej]);
                    } 
                }
            });
        });
    }
    for (var i=0; i < init_nodes.length; i++){
        var gsi = init_nodes[i].group;
        if (selected_components[groups[gsi[0]][0].cid] == false) continue;
        var ni = init_nodes[i].id;
        gsi.forEach(function(gi){
            gsi.forEach(function(gj){
                if (gi != gj) {
                    var ei = gi+n;
                    var ej = gj+n;
                    if (!expand[gi] & !expand[gj]){
                    if (link_set.has([ej,ei]) == false){
                        link_set.add([ei,ej]);
                    } 
                    }
                    if (!expand[gi] & expand[gj]){
                        if (link_set.has([ni,ei]) == false){
                            link_set.add([ei,ni]);
                        } 
                    }
                    if (expand[gi] & !expand[gj]){
                        if (link_set.has([ej,ni]) == false){
                            link_set.add([ni,ej]);
                        } 
                    }
                }
            });
        });
    }
    // console.log(nodes);
    // console.log(node_map);
    link_set.forEach(function (d){
        d = JSON.parse(d);
        links.push({"source":d[0],"target":d[1]});
    });
    // console.log(data.nodes);
    // console.log(links);
    // console.log(nodes);
    // console.log(hulls);
    return {"nodes":nodes, "links":links, "hulls":create_hulls(groups,expand,offset)};
}

function draw_graph(data, init_nodes, graph, expand, groups, is_initial, dblclick_id, group_meta, component_init_pos, selected_components,gDraw,parentWidth,parentHeight){
    graph = update_network(data, init_nodes, graph, expand, groups, is_initial, dblclick_id, group_meta, component_init_pos, selected_components);
    is_initial = false;

    gDraw.selectAll("*").remove();

    // gDraw.selectAll("node").exit().remove();
    // gDraw.selectAll("hull").exit().remove();
    // gDraw.selectAll("link").exit().remove();

    // var width = +svg.attr("width"),
    //     height = +svg.attr("height");

    // var color = d3.scaleOrdinal(d3.schemeCategory10);

    if (! ("links" in graph)) {
        console.log("Graph is missing links");
        return;
    }

    var nodes = {};
    var i;
    for (i = 0; i < graph.nodes.length; i++) {
        nodes[graph.nodes[i].id] = graph.nodes[i];
        graph.nodes[i].weight = 1.01;
    }

    // the brush needs to go before the nodes so that it doesn't
    // get called when the mouse is over a node
    var gBrushHolder = gDraw.append('g');
    var gBrush = null;

    var tooltip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("opacity", 0);

    var simulation = d3.forceSimulation()
        .force("link", d3.forceLink()
                .id(function(d) { return d.id; })
                .distance(function(d) { 
                    if (d.source.node_type=='node' & d.target.node_type=='node'){
                        return 0;
                    }else{
                        return Math.min(Math.max(Math.min(d.source.size,d.target.size),50),200);
                    }
                })
            )
        .force("charge", d3.forceManyBody().strength(-200))
        .velocityDecay(.5)
        // .force("center", d3.forceCenter(parentWidth / 2, parentHeight / 2))
        .force("x", d3.forceX().x(function(d){return component_init_pos[d.cid][0]}))
        .force("y", d3.forceY().y(function(d){return component_init_pos[d.cid][1]}));
    
    gDraw.selectAll("path").remove();
    var hull = gDraw.append("g")
        .attr("class", "hull")
        .selectAll("path")
        .data(graph.hulls)
        .enter().append("path")
        .attr("d", drawCluster)
        .style("fill", "lightblue")
        .attr("opacity", 0.3)
        .on("dblclick", function(d){
            // console.log(d);
            if (expand[d.group[0]]){
                expand[d.group[0]] = !expand[d.group[0]];
                draw_graph(data, init_nodes, graph, expand, groups, is_initial, d.id, group_meta, component_init_pos, selected_components, gDraw,parentWidth,parentHeight);
            }
        });

    var link = gDraw.append("g")
        .attr("class", "link")
        .selectAll("line")
        .data(graph.links)
        .enter().append("line")
        .attr("stroke-width", function(d) { return Math.sqrt(d.value); });

    var node = gDraw.append("g")
        .attr("class", "node")
        .selectAll("g")
        .data(graph.nodes)
        .enter()
        .append("g")
        // .selectAll("circle")
        // .data(graph.nodes)
        // .enter().append("circle")
        // .attr("r", 5)
        // .attr("fill", function(d) { 
        //     if ('color' in d)
        //         return d.color;
        //     else
        //         return color(d.group[0]); 
        // })
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended))
        // .on("click", function(d){
        //     console.log(d);
        // })
        .on("dblclick", function(d){
            tooltip.transition()
                .duration(50)
                .style("opacity", 0);
            // console.log(d);
            if (expand[d.group[0]] == false){
                expand[d.group[0]] = !expand[d.group[0]];
                draw_graph(data, init_nodes, graph, expand, groups, is_initial, d.id, group_meta, component_init_pos, selected_components, gDraw,parentWidth,parentHeight);
            }
        })
        .on("mouseover", function(d, i){
            d3.select(this).transition()
                // .selectAll('circle').attr("stroke","black")
                .duration('50')
                .style("opacity", 0.7);
            tooltip.transition()
                .duration(50)
                .style("opacity", 1);
            // console.log(d3.event.pageX);
            // console.log(d3.event.pageY);
            var num;
            if (d.node_type == "node") {
                num = "Graph node id: "+d.id.toString();
                num += "<br>Prediction: "+d.prediction.toString();
                num += "<br>Truth: "+d.label.toString();
                num += "<br>Component id: "+d.cid.toString();
            }else{
                num = "Reeb node id: "+d.group[0].toString();
                num += "<br>Component id: "+d.cid.toString();
            }
            tooltip.html(num)
               .style("left", (d3.event.pageX + 10) + "px")
               .style("top", (d3.event.pageY - 15) + "px");
        })
        .on("mouseout", function(d, i){
            d3.select(this).transition()
                // .selectAll('circle').attr("stroke","white")
                .duration('50')
                .style("opacity", 1.0);
            tooltip.transition()
                .duration(50)
                .style("opacity", 0);
        });

    /* Draw the respective pie chart for each node */
    node.each(function (d) {
        NodePieBuilder.drawNodePie(d3.select(this), d.pieChart, {
            radius: Math.min(Math.pow(d.size,0.25)*10,50),
            outerStrokeWidth: 3,
            parentNodeColor: "white",
            showPieChartBorder: false,
            showLabelText: false
        });
    });
      
    // add titles for mouseover blurbs
    node.append("title")
        .text(function(d) { 
            if ('name' in d)
                return d.name;
            else
                return d.id; 
        });

    simulation
        .nodes(graph.nodes)
        .on("tick", ticked);

    simulation.force("link")
        .links(graph.links);
    
    
    d3.select("#apply").on("click", function() {
        let text = document.getElementById("cids").value;
        if (text == "all"){
            Object.keys(selected_components).forEach(function(k){
                selected_components[k] = true;
            });
        }else{
            Object.keys(selected_components).forEach(function(k){
                selected_components[k] = false;
            });
            text.split(",").forEach(function(d){
                selected_components[parseInt(d)] = true;
            });
        }
        draw_graph(data, init_nodes, graph, expand, groups, is_initial, null, group_meta, component_init_pos, selected_components, gDraw,parentWidth,parentHeight);
    });

    function ticked() {
        if (graph.hulls.length>0){
            hull.data(create_hulls(groups,expand,offset))
                .attr("d",drawCluster);
        }
        // update node and line positions at every step of 
        // the force simulation
        link.attr("x1", function(d) { return d.source.x; })
            .attr("y1", function(d) { return d.source.y; })
            .attr("x2", function(d) { return d.target.x; })
            .attr("y2", function(d) { return d.target.y; });
        
        d3.selectAll("circle").attr("cx", function (d) {
                return d.x;
            })
            .attr("cy", function (d) {
                return d.y;
            });

        // node.attr("cx", function(d) { return d.x; })
        //     .attr("cy", function(d) { return d.y; });
    }

    // var brushMode = false;
    // var brushing = false;

    // var brush = d3.brush()
    //     .on("start", brushstarted)
    //     .on("brush", brushed)
    //     .on("end", brushended);

    // function brushstarted() {
    //     // keep track of whether we're actively brushing so that we
    //     // don't remove the brush on keyup in the middle of a selection
    //     brushing = true;

    //     node.each(function(d) { 
    //         d.previouslySelected = shiftKey && d.selected; 
    //     });
    // }

    // rect.on('click', () => {
    //     node.each(function(d) {
    //         d.selected = false;
    //         d.previouslySelected = false;
    //         d3.select(this).selectAll('circle').attr("stroke","white");
    //     });
    //     node.classed("selected", false);
    //     // document.getElementById("demo").innerHTML = "";
    // });

    // function brushed() {
    //     if (!d3.event.sourceEvent) return;
    //     if (!d3.event.selection) return;

    //     var extent = d3.event.selection;

    //     node.classed("selected", function(d) {
    //         return d.selected = d.previouslySelected ^
    //         (extent[0][0] <= d.x && d.x < extent[1][0]
    //          && extent[0][1] <= d.y && d.y < extent[1][1]);
    //     });
    //     node.filter(function(d) { return d.selected; })
    //         .each(function(d) { //d.fixed |= 2; 
    //             d3.select(this).selectAll('circle').attr("stroke","black");
    //     })
    // }

    // function brushended() {
    //     if (!d3.event.sourceEvent) return;
    //     if (!d3.event.selection) return;
    //     if (!gBrush) return;

    //     gBrush.call(brush.move, null);

    //     if (!brushMode) {
    //         // the shift key has been release before we ended our brushing
    //         gBrush.remove();
    //         gBrush = null;
    //     }

    //     brushing = false;
    // }

    d3.select('body').on('keydown', keydown);
    d3.select('body').on('keyup', keyup);

    var shiftKey = false;

    function keydown() {
        shiftKey = d3.event.shiftKey;

        // if (shiftKey) {
        //     // if we already have a brush, don't do anything
        //     if (gBrush)
        //         return;

        //     brushMode = true;

        //     if (!gBrush) {
        //         gBrush = gBrushHolder.append('g');
        //         gBrush.call(brush);
        //     }
        // }
    }

    function keyup() {
        shiftKey = false;
        // brushMode = false;

        // if (!gBrush)
        //     return;

        // if (!brushing) {
        //     // only remove the brush if we're not actively brushing
        //     // otherwise it'll be removed when the brushing ends
        //     gBrush.remove();
        //     gBrush = null;
        // }
    }

    function dragstarted(k) {
        // document.getElementById("demo").innerHTML = d.id;
        if (!d3.event.active) simulation.alphaTarget(0.9).restart();

        if(!shiftKey) {
            if (!k.selected && !shiftKey) {
                // if this node isn't selected, then we have to unselect every other node
                node.classed("selected", function(p) { return p.selected =  p.previouslySelected = false; });
            }
            d3.select(this).classed("selected", function(p) { k.previouslySelected = k.selected; return k.selected = true; });

            node.filter(function(d) { return d.selected; })
            .each(function(d) { //d.fixed |= 2; 
            d.fx = d.x;
            d.fy = d.y;
            //   d3.select(this).selectAll('circle').attr("stroke","black");
            });
            // d.dx = 0;
            // d.dy = 0;
        }else{
            node.filter(function(d) { return d.cid == k.cid; })
                .each(function(d) { //d.fixed |= 2; 
                d.fx = d.x;
                d.fy = d.y;
                //   d3.select(this).selectAll('circle').attr("stroke","black");
                });
            k.dx = 0;
            k.dy = 0;
        }
    }

    function dragged(k) {
        if(!shiftKey){
            //d.fx = d3.event.x;
        //d.fy = d3.event.y;
            node.filter(function(d) { return d.selected; })
            .each(function(d) { 
                d.fx += d3.event.dx;
                d.fy += d3.event.dy;
                // d.dx += d3.event.dx;
                // d.dy += d3.event.dy;
            })
        }else{
            node.filter(function(d) { return d.cid == k.cid;})
            .each(function(d) { 
                d.fx += d3.event.dx;
                d.fy += d3.event.dy;
            });
            k.dx += d3.event.dx;
            k.dy += d3.event.dy;
        }
    }

    function dragended(k) {
        if (!d3.event.active) simulation.alphaTarget(0);
        if(!shiftKey){
            k.fx = null;
            k.fy = null;
            node.filter(function(d) { return d.selected; })
            .each(function(d) { //d.fixed &= ~6; 
                d.fx = null;
                d.fy = null;
            })
        }else{
            k.fx = null;
            k.fy = null;
            node.filter(function(d) { return d.cid == k.cid; })
            .each(function(d) { //d.fixed &= ~6; 
                d.fx = null;
                d.fy = null;
            })
            component_init_pos[k.cid] = [
                component_init_pos[k.cid][0]+k.dx,
                component_init_pos[k.cid][1]+k.dy];
            console.log(component_init_pos[k.cid]);
            simulation.force("x").initialize(graph.nodes);
            simulation.force("y").initialize(graph.nodes);
        }
    }

    // var texts = ['Use the scroll wheel to zoom',
    //             //  'Hold the shift key to select nodes',
    //             'Double click a hull to collapse',
    //             'Double click a Reeb net node to expand',
    //             'Click and hold empty area to pan the entire graph',
    //             'Click and hold a node while pressing shift key to pan a single component'];
    // // console.log(parentWidth);
    // // console.log(parentHeight);
    // d3.select('svg').selectAll('text')
    //     .data(texts)
    //     .enter()
    //     .append('text')
    //     .attr('x', parentWidth-10)
    //     .attr('y', function(d,i) { return parentHeight - 19*texts.length + i * 18; })
    //     .text(function(d) { return d; });

    return graph;
}

function createV4SelectableForceDirectedGraph(svg, graph, document) {
    let parentWidth = d3.select('svg').node().parentNode.clientWidth;
    let parentHeight = d3.select('svg').node().parentNode.clientHeight;
    const groups = {}, expand = {}, init_nodes = {}, group_meta = {}, counter = {}, component_init_pos = {}, selected_components = {};
    const data = JSON.parse(JSON.stringify(graph));
    graph.nodes.forEach(function(d){
        d.group.forEach(function(g){
            if (g in groups) {
                groups[g].push(d);
            }else{
                groups[g] = [d];
            }
            expand[g] = false;
            init_nodes[d.id] = d;
        });
        component_init_pos[d.cid] = [0,0];
        selected_components[d.cid] = true;
    });
    var ncomponents = Object.keys(component_init_pos).length;
    var nrows = Math.round(Math.sqrt(ncomponents));
    nrows = Math.min(Math.round(ncomponents/nrows),nrows);
    var block_size = [0.5*parentWidth/nrows,0.5*parentHeight/nrows];
    var center_loc = [0,0];
    Object.keys(component_init_pos).forEach(function(k){
        component_init_pos[k] = [block_size[0]/2+Math.floor(k/nrows)*block_size[0],block_size[1]/2+k%nrows*block_size[1]];
        center_loc[0] += component_init_pos[k][0];
        center_loc[1] += component_init_pos[k][1];
    });
    center_loc[0] /= ncomponents;
    center_loc[1] /= ncomponents;
    Object.keys(component_init_pos).forEach(function(k){
        var curr_loc = component_init_pos[k];
        component_init_pos[k] = [curr_loc[0]-center_loc[0]+parentWidth/2,curr_loc[1]-center_loc[1]+parentHeight/2];
    });
    console.log(component_init_pos);
    Object.keys(groups).forEach(function(k){
        g = groups[k];
        counter[k] = {};
        for (var i=0; i<g.length; i++){
            var l = g[i].prediction;
            if (l in counter[k]){
                counter[k][l] += 1;
            }else{
                counter[k][l] = 1;
            }
        }
        var cnt = g.length;
        var piechart = [];
        Object.keys(counter[k]).forEach(function(d){
            piechart.push({"color":parseInt(d)+1,"percent":100*counter[k][d]/cnt});
        });
        group_meta[k] = piechart;
    });
    // console.log(group_meta);
    const is_initial = true;

    var svg = d3.select('svg')
    .attr('width', parentWidth)
    .attr('height', parentHeight);

    // remove any previous graphs
    // svg.selectAll('.g-main').remove();

    var gMain = svg.append('g')
    .classed('g-main', true);

    var rect = gMain.append('rect')
    .attr('width', parentWidth)
    .attr('height', parentHeight)
    .style('fill', 'transparent');

    var gDraw = gMain.append('g');

    var zoom = d3.zoom()
    .on('zoom', zoomed);

    gMain.call(zoom).on("dblclick.zoom", null);


    function zoomed() {
        gDraw.attr('transform', d3.event.transform);
    }
    
    d3.select("#zoom_in").on("click", function() {
        zoom.scaleBy(gMain.transition().duration(200), 1.2);
    });
    d3.select("#zoom_out").on("click", function() {
        zoom.scaleBy(gMain.transition().duration(200), 0.8);
    });

    return draw_graph(data,init_nodes,graph,expand,groups,is_initial,null,group_meta,component_init_pos,selected_components,gDraw,parentWidth,parentHeight);
};
