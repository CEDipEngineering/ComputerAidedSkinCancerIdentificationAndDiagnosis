import {api} from "../api"

export async function hedImage(uuid){
    return api.get(`hed_images_zoom/${uuid}`).then(
        response => {return response}
    )
}