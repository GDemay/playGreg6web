// Synthetic actual Device flush: translated CPU buffers with contiguous/scattered host slots.
#include "gpu/d3d12_guest_resources.h"
#include "guest/com_bridge.h"
#include "os/guest_window.h"
#include "os/vmem.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace dg=playgta6::gpu::d3d12_guest;
namespace gpu=playgta6::gpu;
namespace com=playgta6::guest::com;
namespace os=playgta6::os;
namespace gw=os::guest_window;
int main(int argc,char** argv) {
 if(argc!=4)return 2;
 const size_t mapped_bytes=std::strtoull(argv[1],nullptr,0)*1024*1024;
 const bool quarter=std::strcmp(argv[3],"fragmented-quarter")==0;
 const bool half=std::strcmp(argv[3],"fragmented-half")==0;
 const bool nearly=std::strcmp(argv[3],"fragmented-nearly-dense")==0;
 const bool dense_resident=quarter||half||nearly||std::strcmp(argv[3],"fragmented-dense-resident")==0;
 const bool dense=dense_resident||std::strcmp(argv[3],"fragmented-dense")==0;
 const bool unaligned=std::strcmp(argv[3],"fragmented-unaligned")==0;
 const size_t bytes=mapped_bytes-(unaligned?246:0);
 const size_t dirty_bytes=quarter?bytes/4:half?bytes/2:nearly?bytes-gw::kSlotBytes:bytes;
 const size_t frames=std::strtoull(argv[2],nullptr,0);
 const bool scattered=dense||unaligned||std::strcmp(argv[3],"fragmented")==0;
 if(!mapped_bytes||mapped_bytes%gw::kSlotBytes||!frames||(!scattered&&std::strcmp(argv[3],"contiguous")))return 2;
 try {
 os::set_write_tracking(false); gw::reset_for_testing();gw::set_enabled(true);
 os::VirtualMemoryArena arena;com::reset_com_bridge();com::set_guest_memory(&arena);
 std::vector<uint8_t> storage(mapped_bytes*2+gw::kSlotBytes,0);
 auto* host=reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(storage.data())+gw::kSlotBytes-1)&~(gw::kSlotBytes-1));
 const uint64_t base=gw::map_runtime_range(host,mapped_bytes,false);
 const uint64_t cpu=base+(unaligned?123:0);
 if(!cpu)throw std::runtime_error("map runtime range failed");
 for(size_t i=0;i<mapped_bytes/gw::kSlotBytes;++i)
  if(!gw::map_slot(base+i*gw::kSlotBytes,host+i*gw::kSlotBytes*(scattered?2:1)))throw std::runtime_error("map slot failed");
 dg::DeviceOptions options;options.sink=[](const uint8_t*,size_t){};
 auto dev=com::create_object<dg::Device>(com::implements(dg::kID3D12Device),options);dev->initialize_guest_payload();
 dg::GuestResourceDesc d{};d.Dimension=1;d.Width=bytes;d.Height=1;d.DepthOrArraySize=1;d.MipLevels=1;d.SampleCount=1;d.Layout=1;
 dg::Resource::Placement place{};place.address=cpu;
 auto r=com::create_object<dg::Resource>(com::implements(dg::kID3D12Resource),dg::device_ref(dev.get()),d,std::move(place),0);
 if(!r||!r->initialize()||r->cpu_address()!=cpu)throw std::runtime_error("buffer init failed");
 auto flush=[&](gpu::CommandEncoder& e){dev->begin_submission();dev->flush_cpu_buffers(e);dev->end_submission();};
 if(dense_resident)for(size_t i=0;i<bytes;i+=4096)if(!com::write_guest(cpu+i,uint32_t{0xA5}))throw std::runtime_error("warm dense write failed");
 {gpu::CommandEncoder e;flush(e);if(e.size()&&!dense_resident)throw std::runtime_error("zero initial buffer uploaded");}
 double ms=0;uint64_t digest=14695981039346656037ull;
 const auto c0=dg::work_total(dg::Work::ShadowComparedBytes);
 for(size_t f=0;f<frames;++f){
  const size_t at=(f%(bytes/gw::kSlotBytes))*gw::kSlotBytes+4096;
  const uint32_t value=static_cast<uint32_t>(f+1);
  if(dense){for(size_t i=0;i<dirty_bytes;i+=4096)if(!com::write_guest(cpu+i,value))throw std::runtime_error("dense write failed");}
  else if(!com::write_guest(cpu+at,value))throw std::runtime_error("write failed");
  gpu::CommandEncoder e;const auto t=std::chrono::steady_clock::now();flush(e);
  ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
  using namespace playgta6;
  size_t cursor=0, uploaded=0;
  const size_t expected_bytes=dense?dirty_bytes:4096;
  while(cursor<e.size()){
   GpuPacketHeader h{};GpuWriteBufferPayload w{};
   if(e.size()-cursor<sizeof(h)+sizeof(w))throw std::runtime_error("short packet");
   std::memcpy(&h,e.data()+cursor,sizeof(h));std::memcpy(&w,e.data()+cursor+sizeof(h),sizeof(w));
   const size_t n=std::min<size_t>(1<<20,expected_bytes-uploaded);
   if(h.opcode!=static_cast<uint16_t>(GpuOpcode::WriteBuffer)||w.id!=r->id()||w.dst_offset!=(dense?uploaded:at)||w.data_len!=n||h.payload_bytes!=sizeof(w)+n||e.size()-cursor<sizeof(h)+sizeof(w)+n)throw std::runtime_error("wrong packet");
   for(size_t k=0;k<n;k+=4){uint32_t word;std::memcpy(&word,e.data()+cursor+sizeof(h)+sizeof(w)+k,4);if(word!=(k%4096?0:value))throw std::runtime_error("wrong payload");}
   cursor+=sizeof(h)+sizeof(w)+n;uploaded+=n;
  }
  if(uploaded!=expected_bytes)throw std::runtime_error("wrong uploaded bytes");
  for(size_t k=0;k<e.size();++k){digest^=e.data()[k];digest*=1099511628211ull;}
 }
 // Untimed short-tail and unreadable-slot regression controls (packet byte checks above
 // remain the measured sparse-update differential).
 const uint64_t compared=dg::work_total(dg::Work::ShadowComparedBytes)-c0;
 if(unaligned){
  if(!com::write_guest(cpu+bytes-1,uint8_t{77}))throw std::runtime_error("tail write failed");
  gpu::CommandEncoder e;flush(e);playgta6::GpuWriteBufferPayload w{};
  if(e.size()<sizeof(playgta6::GpuPacketHeader)+sizeof(w))throw std::runtime_error("tail missing");
  std::memcpy(&w,e.data()+sizeof(playgta6::GpuPacketHeader),sizeof(w));
  if(w.dst_offset!=(bytes/4096)*4096||w.data_len!=bytes%4096||e.data()[e.size()-1]!=77)throw std::runtime_error("tail wrong");
  // Verification must fully initialize/read fragmented staging, including an unmarked page.
  const uint64_t misses=gw::verify_misses();
  *static_cast<uint8_t*>(gw::translate(cpu+99))=55;
  gw::set_write_marks_verified(true);
  gpu::CommandEncoder verified;flush(verified);
  if(verified.size()!=sizeof(playgta6::GpuPacketHeader)+sizeof(w)+4096)throw std::runtime_error("verification packet missing");
  std::memcpy(&w,verified.data()+sizeof(playgta6::GpuPacketHeader),sizeof(w));
  if(w.dst_offset!=0||w.data_len!=4096||verified.data()[sizeof(playgta6::GpuPacketHeader)+sizeof(w)+99]!=55||gw::verify_misses()!=misses+1)throw std::runtime_error("fragmented verification failed");
  gw::set_write_marks_verified(false);
  // Direct/null and unscanned callers retain the original full staging path.
  for(unsigned direct=0;direct<2;++direct){
   if(!com::write_guest(cpu+100,static_cast<uint8_t>(66+direct)))throw std::runtime_error("direct write failed");
   gpu::CommandEncoder out;gw::MarkPass unscanned;unscanned.begin();
   if(r->flush_shadow(out,direct?&unscanned:nullptr)!=4096||out.size()!=sizeof(playgta6::GpuPacketHeader)+sizeof(w)+4096)throw std::runtime_error("direct flush failed");
   std::memcpy(&w,out.data()+sizeof(playgta6::GpuPacketHeader),sizeof(w));
   if(w.dst_offset!=0||w.data_len!=4096||out.data()[sizeof(playgta6::GpuPacketHeader)+sizeof(w)+100]!=66+direct)throw std::runtime_error("direct payload failed");
  }
  if(!gw::unmap_slot(base+mapped_bytes-gw::kSlotBytes))throw std::runtime_error("unmap failed");
  gpu::CommandEncoder hole;flush(hole);if(hole.size())throw std::runtime_error("unmapped page changed");
 }
 std::printf("{\"bytes\":%zu,\"written_bytes_per_flush\":%zu,\"flushes\":%zu,\"mapping\":\"%s\",\"wall_ms\":%.9f,\"compared_bytes\":%llu,\"digest\":\"%016llx\",\"correct\":true}\n",bytes,dense?dirty_bytes:4096,frames,argv[3],ms,(unsigned long long)compared,(unsigned long long)digest);
 r.reset();dev.reset();com::reset_com_bridge();com::set_guest_memory(nullptr);gw::reset_for_testing();
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
