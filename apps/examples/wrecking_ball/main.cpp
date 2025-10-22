#include <catch2/catch_all.hpp>
#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/common/timer.h>
#include <filesystem>
#include <fstream>
#include <numbers>


template <typename T>
bool convertJsonParameters(nlohmann::json j, T& param)
{
   try
   {
       param = j.get<T>();
   }
   catch(nlohmann::json::exception& e)
   {
       std::cout << e.what() << "\n";
       return false;
   }
   return true;
}

int main()
{
   using namespace uipc;
   using namespace uipc::core;
   using namespace uipc::geometry;
   using namespace uipc::constitution;
   namespace fs = std::filesystem;

   Timer global_timer("main");
   global_timer.enable_all();
   spdlog::set_level(spdlog::level::info);

   std::string tetmesh_dir{AssetDir::tetmesh_path()};
   auto        this_output_path = AssetDir::output_path(__FILE__);
   auto        this_folder      = AssetDir::folder(__FILE__);


   Engine engine{"cuda", this_output_path};
   World  world{engine};

   auto config                             = Scene::default_config();
   config["gravity"]                       = Vector3{0, -9.8, 0};
   config["contact"]["friction"]["enable"] = true;
   config["contact"]["enable"]             = true;
   config["contact"]["d_hat"]              = 0.01;
   config["line_search"]["max_iter"]       = 8;

   {  // dump config
       std::ofstream ofs(fmt::format("{}config.json", this_output_path));
       ofs << config.dump(4);
   }


   Scene scene{config};
   {
       Json wrecking_ball_scene;
       {
           std::ifstream ifs(fmt::format("{}wrecking_ball.json", this_folder));
           ifs >> wrecking_ball_scene;
       }

       Json wrecking_ball_vertexColoring;
       {
           std::ifstream ifs(fmt::format("{}model.VertexColoring.json", this_folder));
           ifs >> wrecking_ball_vertexColoring;
       }

       Json wrecking_ball_edgeColoring;
       {
           std::ifstream ifs(fmt::format("{}model.EdgeColoring.json", this_folder));
           ifs >> wrecking_ball_edgeColoring;
       }

       Json wrecking_ball_tetColoring;
       //{
       //    std::ifstream ifs(fmt::format("{}model.TetColoring.json", this_folder));
       //    ifs >> wrecking_ball_tetColoring;
       //}

       std::vector<std::vector<int32_t>> verticesColoringCategories;
       std::vector<std::vector<int32_t>> edgesColoringCategories;
       std::vector<std::vector<int32_t>> tetsColoringCategories;
       convertJsonParameters(wrecking_ball_vertexColoring, verticesColoringCategories);
       convertJsonParameters(wrecking_ball_edgeColoring, edgesColoringCategories);
       convertJsonParameters(wrecking_ball_tetColoring, tetsColoringCategories);
       // we sort them from large to small for the aggregated solve
       std::sort(tetsColoringCategories.begin(),
                 tetsColoringCategories.end(),
                 [](const std::vector<int>& a, const std::vector<int>& b)
                 { return a.size() > b.size(); });
       // we sort them from large to small for the aggregated solve
       std::sort(tetsColoringCategories.begin(),
                 tetsColoringCategories.end(),
                 [](const std::vector<int>& a, const std::vector<int>& b)
                 { return a.size() > b.size(); });

       ////#######################verticesColoringCategories this info need to be saved inside
       ////#######################call it when needed 
       // 
       // 
       // create constitution and contact model
       AffineBodyConstitution abd;
       scene.constitution_tabular().insert(abd);
       auto default_contact = scene.contact_tabular().default_element();
       scene.contact_tabular().default_model(0.01, 20.0_GPa);

       Float     scale = 1;
       Transform T     = Transform::Identity();
       T.scale(scale);
       SimplicialComplexIO io{T};

       auto cube = io.read(fmt::format("{}cube.msh", tetmesh_dir));
       auto ball = io.read(fmt::format("{}ball.msh", tetmesh_dir));
       auto link = io.read(fmt::format("{}link.msh", tetmesh_dir));

       S<Object> cube_obj = scene.objects().create("cubes");
       S<Object> ball_obj = scene.objects().create("balls");
       S<Object> link_obj = scene.objects().create("links");

       abd.apply_to(cube, 10.0_MPa);
       label_surface(cube);
       label_triangle_orient(cube);
       abd.apply_to(ball, 10.0_MPa);
       label_surface(ball);
       label_triangle_orient(ball);
       abd.apply_to(link, 10.0_MPa);
       label_surface(link);
       label_triangle_orient(link);

       default_contact.apply_to(cube);
       default_contact.apply_to(ball);
       default_contact.apply_to(link);


       auto build_mesh = [&](const Json& j, Object& obj, const SimplicialComplex& mesh)
       {
           Vector3 position;

           if(j.find("position") != j.end())
           {
               position[0] = j["position"][0].get<Float>();
               position[1] = j["position"][1].get<Float>();
               position[2] = j["position"][2].get<Float>();
           }

           Eigen::Quaternion<Float> Q = Eigen::Quaternion<Float>::Identity();

           if(j.find("rotation") != j.end())
           {
               Vector3 rotation;
               rotation[0] = j["rotation"][0].get<Float>();
               rotation[1] = j["rotation"][1].get<Float>();
               rotation[2] = j["rotation"][2].get<Float>();

               rotation *= std::numbers::pi / 180.0;

               Q = AngleAxis(rotation.z(), Vector3::UnitZ())
                   * AngleAxis(rotation.y(), Vector3::UnitY())
                   * AngleAxis(rotation.x(), Vector3::UnitX());
           }

           IndexT is_fixed = 0;
           if(j.find("is_dof_fixed") != j.end())
           {
               is_fixed = j["is_dof_fixed"].get<bool>() ? 1 : 0;
           }

           position *= scale;

           Transform t = Transform::Identity();
           t.translate(position).rotate(Q);

           SimplicialComplex this_mesh     = mesh;
           view(this_mesh.transforms())[0] = t.matrix();

           auto is_fixed_attr = this_mesh.instances().find<IndexT>(builtin::is_fixed);
           view(*is_fixed_attr)[0] = is_fixed;


           obj.geometries().create(this_mesh);

           this_mesh.verticesColoringCategories = verticesColoringCategories;
           this_mesh.edgesColoringCategories    = edgesColoringCategories;
           this_mesh.tetsColoringCategories     = tetsColoringCategories;
       };

       //IndexT count = 0;

       for(const Json& obj : wrecking_ball_scene)
       {
           if(obj["mesh"] == "link.msh")
           {
               build_mesh(obj, *link_obj, link);
           }
           else if(obj["mesh"] == "cube.msh")
           {
               build_mesh(obj, *cube_obj, cube);
           }
           else if(obj["mesh"] == "ball.msh")
           {
               build_mesh(obj, *ball_obj, ball);
           }
       }


       constexpr bool UseMeshGround = false;

       if(UseMeshGround)
       {
           Transform pre_transform = Transform::Identity();
           pre_transform.scale(Vector3{40, 0.2, 40});

           SimplicialComplexIO io{pre_transform};
           io          = SimplicialComplexIO{pre_transform};
           auto ground = io.read(fmt::format("{}{}", tetmesh_dir, "cube.msh"));

           label_surface(ground);
           label_triangle_orient(ground);

           Transform transform = Transform::Identity();
           transform.translate(Vector3{12, -1.1, 0});
           view(ground.transforms())[0] = transform.matrix();
           abd.apply_to(ground, 10.0_MPa);

           auto is_fixed = ground.instances().find<IndexT>(builtin::is_fixed);
           view(*is_fixed)[0] = 1;

           auto ground_obj = scene.objects().create("ground");
           ground_obj->geometries().create(ground);
       }
       else
       {
           auto ground_obj = scene.objects().create("ground");
           auto g          = geometry::ground(-1.0);
           ground_obj->geometries().create(g);
       }
   }

   world.init(scene);
   SceneIO sio{scene};
   sio.write_surface(fmt::format("{}scene_surface{}.obj", this_output_path, 0));

   auto obj_all = scene.objects().find("balls")[0];
   auto ids  = obj_all->geometries().ids();

   auto geo = scene.geometries()
                  .find(ids[0])
                  .rest_geometry->geometry()
                  .as<SimplicialComplex>();

   
   while(world.frame() < 300)
   {
       world.advance();
       world.retrieve();
       //world.dump();

       sio.write_surface(
           fmt::format("{}scene_surface{}.obj", this_output_path, world.frame()));
       // fmt::println("frame: {}", world.frame());
   }
   global_timer.report();
}